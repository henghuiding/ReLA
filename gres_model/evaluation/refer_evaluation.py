import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict, defaultdict
import torch

from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U

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
            output_mask = output["ref_seg"].argmax(dim=0).to(self._cpu_device)
            pred_mask = np.array(output_mask, dtype=np.int8)
            gt_mask_data = input.get('gt_mask_merged')
            if isinstance(gt_mask_data, torch.Tensor):
                gt_mask_data = gt_mask_data.to(self._cpu_device).numpy()
            elif isinstance(gt_mask_data, np.ndarray):
                pass
            elif gt_mask_data is None:
                if not self._warned_missing_gt_mask:
                    self._logger.warning(
                        "[ReferEvaluator] Missing 'gt_mask_merged'; using zeros"
                    )
                    self._warned_missing_gt_mask = True
                height = input.get('height') or pred_mask.shape[0]
                width = input.get('width') or (
                    pred_mask.shape[1] if pred_mask.ndim > 1 else pred_mask.shape[0]
                )
                gt_mask_data = np.zeros((height, width), dtype=np.uint8)
            else:
                gt_mask_data = np.array(gt_mask_data, dtype=np.uint8)
            gt = np.array(gt_mask_data, dtype=np.int8)

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
                'gt_nt': input.get('empty', False),
                'pred_mask': pred_mask,
                'gt_mask': gt
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
                # False Positive
                if eval_sample['pred_nt']:
                    nt[src]["FP"] += 1
                    I = 0

                # True Negative
                else:
                    nt[src]["TN"] += 1

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

            total_count[src] += 1
            results_dict.append(ref_result)

        detected_srcs = [src for src in self._available_sources if total_count[src] > 0]

        final_results_list = []
        
        # results for each source
        for src in detected_srcs:
            res = {}
            res['gIoU'] = 100. * (accum_IoU[src] / total_count[src])
            res['cIoU'] = accum_I[src] * 100. / accum_U[src]

            self._logger.info(str(nt[src]))
            if empty_count[src] > 0:
                res['T_acc'] = nt[src]['TN'] / (nt[src]['TN'] + nt[src]['FP'])
                res['N_acc'] = nt[src]['TP'] / (nt[src]['TP'] + nt[src]['FN'])
            else:
                res['T_acc'] = res['N_acc'] = 0

            for thres in pr_thres:
                pr_name = 'Pr@{0:1.1f}'.format(thres)
                res[pr_name] = pr_count[src][thres] * 100. / not_empty_count[src]
            
            final_results_list.append((src, res))
        
        def _sum_values(x):
            return sum(x.values())
        
        # global results
        if len(detected_srcs) > 1:
            res_full = {}
            res_full['gIoU'] = 100. * _sum_values(accum_IoU) / _sum_values(total_count)
            res_full['cIoU'] =  100. * _sum_values(accum_I) / _sum_values(accum_U)

            for thres in pr_thres:
                pr_name = 'Pr@{0:1.1f}'.format(thres)
                res_full[pr_name] = sum([pr_count[src][thres] for src in detected_srcs]) * 100. / _sum_values(not_empty_count)

            final_results_list.append(('full', res_full))
        
        if self._output_dir:
            file_path = os.path.join(self._output_dir, f"{self._dataset_name}_results.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(final_results_list, indent=4))

            file_path = os.path.join(self._output_dir, f"{self._dataset_name}_detailed_results.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(results_dict, indent=4))

        results = OrderedDict(final_results_list)
        self._logger.info(results)
        return results
