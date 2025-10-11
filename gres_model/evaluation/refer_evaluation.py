import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict, defaultdict
import torch
import torch.nn.functional as F

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import BoxMode


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def _to_uint8_mask(mask):
    if mask is None:
        return None
    if torch.is_tensor(mask):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    if mask_np.size == 0:
        return None

    mask_np = mask_np.astype(np.uint8, copy=False)
    if mask_np.ndim >= 3:
        mask_np = mask_np.reshape(mask_np.shape[-2], mask_np.shape[-1])
    return mask_np


def _resize_mask(mask, height, width):
    if mask is None:
        return None
    if mask.shape == (height, width):
        return mask
    if cv2 is not None:
        return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    tensor = torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(height, width), mode="nearest")
    return resized.squeeze(0).squeeze(0).to(dtype=torch.uint8).cpu().numpy()


def _mask_from_boxes(annotations, height, width):
    if not annotations:
        return None
    bbox_mask = np.zeros((height, width), dtype=np.uint8)
    has_box = False
    for ann in annotations:
        bbox = ann.get("bbox")
        if bbox is None:
            continue
        bbox_mode = ann.get("bbox_mode", BoxMode.XYXY_ABS)
        x0, y0, x1, y1 = BoxMode.convert(bbox, bbox_mode, BoxMode.XYXY_ABS)
        x0 = max(0, min(int(np.floor(x0)), width))
        y0 = max(0, min(int(np.floor(y0)), height))
        x1 = max(x0, min(int(np.ceil(x1)), width))
        y1 = max(y0, min(int(np.ceil(y1)), height))
        if x1 <= x0 or y1 <= y0:
            continue
        bbox_mask[y0:y1, x0:x1] = 1
        has_box = True
    if not has_box:
        return None
    return bbox_mask

class ReferEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        save_imgs=False,
    ):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._save_imgs = save_imgs

        self._cpu_device = torch.device("cpu")
        self._warned_missing_source = False
        self._warned_missing_gt_mask = False

        default_sources = [
            "refcoco",
            "grefcoco",
            "unc",
            "google",
            "refcocog",
            "miami2025",
        ]
        # maintain deterministic order while supporting fast membership lookups
        self._available_sources = list(dict.fromkeys(default_sources))
        self._known_sources = set(self._available_sources)

        self._num_classes = 2

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):

            img_id = input['image_id']
            src = input.get('source')
            if not src:
                if not self._warned_missing_source:
                    self._logger.warning(
                        "[ReferEvaluator] Missing 'source' in inputs; defaulting to 'miami2025'"
                    )
                    self._warned_missing_source = True
                src = "miami2025"
            if src not in self._known_sources:
                self._known_sources.add(src)
                self._available_sources.append(src)
                self._logger.warning(
                    f"[ReferEvaluator] Unknown source '{src}' seen; added dynamically."
                )

            # output mask
            output_mask = output["ref_seg"].argmax(dim=0)
            output_mask_cpu = output_mask.to(self._cpu_device)

            def _safe_dim(value, fallback):
                try:
                    dim = int(value)
                except (TypeError, ValueError):
                    dim = 0
                return dim if dim > 0 else fallback

            mask_height = output_mask_cpu.shape[0] if output_mask_cpu.ndim >= 1 else 1
            mask_width = output_mask_cpu.shape[1] if output_mask_cpu.ndim >= 2 else mask_height

            height = _safe_dim(input.get("height"), mask_height)
            width = _safe_dim(input.get("width"), mask_width)

            pred_mask = _to_uint8_mask(output_mask_cpu)
            annotations = input.get("annotations")
            gt_mask_data = input.get('gt_mask_merged')
            gt_mask_source = "gt_mask_merged"
            mask_status = input.get("mask_status") or {}
            gt_mask = _to_uint8_mask(gt_mask_data)

            if gt_mask is None:
                gt_mask = _mask_from_boxes(annotations, height, width)
                gt_mask_source = "bbox_fallback"
                if gt_mask is None:
                    if not self._warned_missing_gt_mask:
                        self._logger.warning(
                            "[ReferEvaluator] Missing 'gt_mask_merged' and bbox fallback for img_id %s",
                            img_id,
                        )
                        self._warned_missing_gt_mask = True
                    gt_mask = np.zeros((height, width), dtype=np.uint8)
                    gt_mask_source = "missing"

            gt_mask = _resize_mask(gt_mask, height, width)
            if gt_mask is None:
                gt_mask = np.zeros((height, width), dtype=np.uint8)

            if pred_mask is None:
                pred_mask = np.zeros((height, width), dtype=np.uint8)
            else:
                pred_mask = _resize_mask(pred_mask, gt_mask.shape[0], gt_mask.shape[1])

            pred_mask = (pred_mask > 0).astype(np.uint8)
            gt_mask = (gt_mask > 0).astype(np.uint8)

            gt_nt_flag = bool(input.get('empty', False))
            mask_valid = True
            skip_reason = None
            skip_iou = False

            if gt_nt_flag:
                skip_iou = True
                skip_reason = 'no_target'

            if not gt_nt_flag and gt_mask.sum() == 0:
                mask_valid = False
                skip_reason = 'empty_gt_mask'
                self._logger.warning(
                    "[ReferEvaluator] Empty GT mask for targeted sample img_id=%s; skipping IoU.",
                    img_id,
                )

            if gt_mask_source == "bbox_fallback" and gt_mask.sum() == 0:
                self._logger.warning(
                    "[ReferEvaluator] Bbox fallback produced empty mask for img_id=%s.",
                    img_id,
                )

            self._logger.debug(
                "[ReferEvaluator] Mask alignment img_id=%s pred_shape=%s gt_shape=%s source=%s",
                img_id,
                pred_mask.shape,
                gt_mask.shape,
                gt_mask_source,
            )

            # output NT label
            output_nt = output["nt_label"].argmax(dim=0).bool().to(self._cpu_device)
            pred_nt = bool(output_nt)

            sentence_info = input.get('sentence')
            sent_text = ""
            if isinstance(sentence_info, dict):
                sent_text = (
                    sentence_info.get('raw')
                    or sentence_info.get('sent')
                    or ""
                )
            elif isinstance(sentence_info, str):
                sent_text = sentence_info
            elif sentence_info is not None:
                sent_text = str(sentence_info)

            sent_info_payload = input.get('sentence_info', sentence_info)
            if isinstance(sent_info_payload, str):
                sent_info_payload = {"raw": sent_info_payload}

            self._predictions.append({
                'img_id': img_id,
                'source': src,
                'sent': sent_text,
                'sent_info': sent_info_payload,
                'pred_nt': pred_nt,
                'gt_nt': gt_nt_flag,
                'pred_mask': pred_mask,
                'gt_mask': gt_mask,
                'mask_valid': mask_valid,
                'skip_iou': skip_iou,
                'skip_reason': skip_reason,
                'gt_mask_source': gt_mask_source,
                'pred_shape': tuple(pred_mask.shape),
                'gt_shape': tuple(gt_mask.shape),
                'gt_mask_sum': int(gt_mask.sum()),
                'mask_status': mask_status,
                'height': height,
                'width': width
                })

    def evaluate(self):
        if self._distributed:
            synchronize()
            predictions = all_gather(self._predictions)
            predictions = list(itertools.chain(*predictions))
            if not is_main_process():
                return
        else:
            predictions = self._predictions

        if self._output_dir and self._save_imgs:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "ref_seg_predictions.pth")
            self._logger.info(f'Saving output images to {file_path} ...')
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)
        
        pr_thres = [.7, .8, .9]

        accum_I = defaultdict(float)
        accum_U = defaultdict(float)
        accum_IoU = defaultdict(float)
        pr_count = defaultdict(lambda: {thres: 0 for thres in pr_thres})
        total_count = defaultdict(int)
        not_empty_count = defaultdict(int)
        empty_count = defaultdict(int)
        nt = defaultdict(lambda: {"TP": 0, "TN": 0, "FP": 0, "FN": 0})

        for src in list(self._available_sources):
            # touch every key to keep deterministic ordering for existing sources
            _ = accum_I[src]
            _ = accum_U[src]
            _ = accum_IoU[src]
            pr_count[src] = {thres: 0 for thres in pr_thres}
            total_count[src] = 0
            not_empty_count[src] = 0
            empty_count[src] = 0
            nt[src] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

        results_dict = []
        decode_stats = defaultdict(int)

        for eval_sample in predictions:
            src = eval_sample['source']
            if src not in self._known_sources:
                self._known_sources.add(src)
                self._available_sources.append(src)
                self._logger.warning(
                    f"[ReferEvaluator] Unknown source '{src}' seen during evaluation; added dynamically."
                )

            ref_result = {}
            ref_result['source'] = src
            ref_result['img_id'] = eval_sample['img_id']
            ref_result['gt_nt'] = eval_sample['gt_nt']
            ref_result['pred_nt'] = eval_sample['pred_nt']
            ref_result['sent'] = eval_sample['sent']
            ref_result['sent_info'] = eval_sample['sent_info']
            ref_result['gt_mask_source'] = eval_sample.get('gt_mask_source')
            ref_result['pred_shape'] = eval_sample.get('pred_shape')
            ref_result['gt_shape'] = eval_sample.get('gt_shape')
            ref_result['gt_mask_sum'] = eval_sample.get('gt_mask_sum')
            ref_result['mask_status'] = eval_sample.get('mask_status')

            mask_status = eval_sample.get('mask_status') or {}
            for group in mask_status.get('groups', []):
                status_str = str(group.get('status', '')).lower()
                if 'rle' in status_str:
                    decode_stats['rle'] += 1
                if 'poly' in status_str:
                    decode_stats['poly'] += 1
                if 'bbox' in status_str:
                    decode_stats['bbox'] += 1
                if 'synthetic' in status_str:
                    decode_stats['synthetic'] += 1
            if mask_status.get('fallback_used'):
                decode_stats['fallback_groups'] += 1

            if not eval_sample.get('mask_valid', True):
                skip_reason = eval_sample.get('skip_reason') or 'invalid_mask'
                self._logger.warning(
                    "[ReferEvaluator] Skipping IoU for img_id=%s (reason: %s).",
                    eval_sample['img_id'],
                    skip_reason,
                )
                ref_result['skip_iou'] = True
                ref_result['skip_reason'] = skip_reason
                results_dict.append(ref_result)
                continue

            skip_iou = bool(eval_sample.get('skip_iou', False))
            ref_result['skip_iou'] = skip_iou
            ref_result['skip_reason'] = eval_sample.get('skip_reason')

            if skip_iou:
                I = 0
                U = 0
            else:
                I, U = computeIoU(eval_sample['pred_mask'], eval_sample['gt_mask'])

            # No-target Samples
            if eval_sample['gt_nt']:
                empty_count[src] += 1
                ref_result['I'] = int(0)

                # True Positive
                if eval_sample['pred_nt']:
                    nt[src]["TP"] += 1
                    accum_IoU[src] += 1
                    accum_I[src] += 0
                    accum_U[src] += 0

                    ref_result['U'] = int(0)
                    ref_result['cIoU'] = float(1)

                # False Negative
                else:
                    nt[src]["FN"] += 1
                    accum_IoU[src] += 0
                    accum_I[src] += 0
                    accum_U[src] += int(U)

                    ref_result['U'] = int(U)
                    ref_result['cIoU'] = float(0)

            # Targeted Samples
            else:
                if eval_sample['pred_nt']:
                    nt[src]["FP"] += 1
                    I = 0
                else:
                    nt[src]["TN"] += 1

                if skip_iou:
                    ref_result['I'] = int(0)
                    ref_result['U'] = int(0)
                    ref_result['cIoU'] = float(0)
                else:
                    this_iou = float(0) if U == 0 else float(I) / float(U)

                    accum_IoU[src] += this_iou
                    accum_I[src] += I
                    accum_U[src] += U

                    not_empty_count[src] += 1

                    for thres in pr_thres:
                        if this_iou >= thres:
                            pr_count[src][thres] += 1

                    ref_result['I'] = int(I)
                    ref_result['U'] = int(U)
                    ref_result['cIoU'] = float(this_iou)
                if skip_iou:
                    # ensure targeted samples without IoU do not pollute denominators
                    ref_result['I'] = int(0)
                    ref_result['U'] = int(0)
                    ref_result['cIoU'] = float(0)

            total_count[src] += 1
            results_dict.append(ref_result)

        detected_srcs = [src for src in self._available_sources if total_count[src] > 0]

        final_results_list = []
        
        # results for each source
        for src in detected_srcs:
            res = {}
            total = total_count[src]
            res['gIoU'] = 100. * (accum_IoU[src] / total) if total > 0 else 0.0
            union = accum_U[src]
            res['cIoU'] = accum_I[src] * 100. / union if union > 0 else 0.0

            self._logger.info(str(nt[src]))
            if empty_count[src] > 0:
                res['T_acc'] = nt[src]['TN'] / (nt[src]['TN'] + nt[src]['FP'])
                res['N_acc'] = nt[src]['TP'] / (nt[src]['TP'] + nt[src]['FN'])
            else:
                res['T_acc'] = res['N_acc'] = 0

            for thres in pr_thres:
                pr_name = 'Pr@{0:1.1f}'.format(thres)
                denom = not_empty_count[src]
                res[pr_name] = pr_count[src][thres] * 100. / denom if denom > 0 else 0.0
            
            final_results_list.append((src, res))
        
        def _sum_values(x):
            return sum(x.values())
        
        # global results
        if len(detected_srcs) > 1:
            res_full = {}
            total_sum = _sum_values(total_count)
            res_full['gIoU'] = 100. * _sum_values(accum_IoU) / total_sum if total_sum > 0 else 0.0
            union_sum = _sum_values(accum_U)
            res_full['cIoU'] =  100. * _sum_values(accum_I) / union_sum if union_sum > 0 else 0.0

            for thres in pr_thres:
                pr_name = 'Pr@{0:1.1f}'.format(thres)
                denom = _sum_values(not_empty_count)
                res_full[pr_name] = sum([pr_count[src][thres] for src in detected_srcs]) * 100. / denom if denom > 0 else 0.0

            final_results_list.append(('full', res_full))
        
        if self._output_dir:
            file_path = os.path.join(self._output_dir, f"{self._dataset_name}_results.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(final_results_list, indent=4))

            file_path = os.path.join(self._output_dir, f"{self._dataset_name}_detailed_results.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(results_dict, indent=4))

        if decode_stats:
            self._logger.info("[ReferEvaluator] Mask decode stats: %s", dict(decode_stats))

        results = OrderedDict(final_results_list)
        self._logger.info(results)
        return results
