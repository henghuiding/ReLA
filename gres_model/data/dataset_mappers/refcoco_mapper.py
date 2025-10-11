import time
import copy
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

from transformers import BertTokenizer
from pycocotools import mask as coco_mask

from gres_model.utils.mask_ops import (
    _poly_to_mask_safe,
    _rle_to_mask_safe,
    bbox_to_mask,
    merge_instance_masks,
)

logger = logging.getLogger(__name__)

__all__ = ["RefCOCOMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_train(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE

    augmentation = []

    augmentation.extend([
        T.Resize((image_size, image_size))
    ])

    return augmentation


def build_transform_test(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE

    augmentation = []

    augmentation.extend([
        T.Resize((image_size, image_size))
    ])

    return augmentation


def _infer_bbox_from_segmentation(obj):
    seg = obj.get("segmentation", [])
    if not seg:
        return None
    xs = []
    ys = []
    for poly in seg:
        if not poly or len(poly) < 4:
            continue
        xs.extend(poly[0::2])
        ys.extend(poly[1::2])
    if not xs or not ys:
        return None
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)
    if x1 <= x0 or y1 <= y0:
        return None
    return [float(x0), float(y0), float(x1), float(y1)]


# This is specifically designed for the COCO dataset.
class RefCOCOMapper:
    _INSTANCE_DATA_CACHE = {}

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        bert_type,
        max_tokens,
        merge=True
    ):
        self.is_train = is_train
        self.merge = merge
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "Full TransformGens used: {}".format(str(self.tfm_gens))
        )

        self.bert_type = bert_type
        self.max_tokens = max_tokens
        logging.getLogger(__name__).info(
            "Loading BERT tokenizer: {}...".format(self.bert_type)
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_type)

        self.img_format = image_format
        self._warned_missing_inst_json = False

        default_inst_path = "/autodl-tmp/rela_data/annotations/instances.json"
        self.id_to_ann = {}
        if os.path.exists(default_inst_path):
            try:
                with open(default_inst_path, "r", encoding="utf-8") as f:
                    inst_data = json.load(f)
                anns = inst_data.get("annotations", []) or []
                self.id_to_ann = {
                    int(ann["id"]): ann
                    for ann in anns
                    if isinstance(ann, dict) and "id" in ann
                }
                print(
                    f"[RefCOCOMapper] Preloaded {len(self.id_to_ann)} instance annotations from {default_inst_path}"
                )
            except (OSError, ValueError, TypeError) as exc:
                logger.warning(
                    "[RefCOCOMapper] Failed to preload default instances json %s: %s",
                    default_inst_path,
                    exc,
                )
                self.id_to_ann = {}
        else:
            print(
                f"[RefCOCOMapper] Warning: default instances file not found at {default_inst_path}"
            )

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if is_train:
            tfm_gens = build_transform_train(cfg)
        else:
            tfm_gens = build_transform_test(cfg)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "bert_type": cfg.REFERRING.BERT_TYPE,
            "max_tokens": cfg.REFERRING.MAX_TOKENS,
        }
        return ret

    @staticmethod
    def _merge_masks(x):
        return x.sum(dim=0, keepdim=True).clamp(max=1)

    @staticmethod
    def _normalize_hw(height, width, fallback_shape):
        def _safe_int(value, default=0):
            try:
                value_i = int(value)
            except (TypeError, ValueError):
                value_i = default
            return value_i if value_i > 0 else default

        h = _safe_int(height)
        w = _safe_int(width)

        fallback_h = 0
        fallback_w = 0
        if fallback_shape is not None:
            if len(fallback_shape) >= 1:
                fallback_h = _safe_int(fallback_shape[0])
            if len(fallback_shape) >= 2:
                fallback_w = _safe_int(fallback_shape[1])

        h = h or fallback_h or 1
        w = w or fallback_w or 1
        return h, w

    @staticmethod
    def _decode_annotation_mask(ann, height, width, *, original_hw=None):
        seg = ann.get("segmentation")
        masks = []
        statuses = []
        original_size = None
        if original_hw:
            try:
                oh, ow = int(original_hw[0]), int(original_hw[1])
                if oh > 0 and ow > 0:
                    original_size = (oh, ow)
            except (TypeError, ValueError):
                original_size = None

        if isinstance(seg, dict):
            mask, status = _rle_to_mask_safe(seg, height, width, original_size=original_size)
            if mask is not None:
                masks.append(mask)
            statuses.append(status)
        elif isinstance(seg, (list, tuple)):
            if not seg:
                statuses.append("seg_missing")
            elif all(isinstance(poly, (list, tuple)) for poly in seg):
                mask, status = _poly_to_mask_safe(seg, height, width, original_size=original_size)
                if mask is not None:
                    masks.append(mask)
                statuses.append(status)
            else:
                for piece in seg:
                    if isinstance(piece, dict):
                        piece_mask, piece_status = _rle_to_mask_safe(
                            piece, height, width, original_size=original_size
                        )
                    elif isinstance(piece, (list, tuple)):
                        piece_mask, piece_status = _poly_to_mask_safe(
                            [piece], height, width, original_size=original_size
                        )
                    else:
                        piece_mask, piece_status = (None, "seg_unsupported")
                    if piece_mask is not None:
                        masks.append(piece_mask)
                    statuses.append(piece_status)
        else:
            statuses.append("seg_missing")

        return merge_instance_masks(masks, height, width, statuses=statuses)

    @staticmethod
    def _collect_ann_ids(dataset_dict, raw_annotations):
        ann_ids = []
        candidates = []
        for key in ("ann_ids", "ann_id", "ann_id_list", "annotation_ids"):
            value = dataset_dict.get(key)
            if isinstance(value, (list, tuple)):
                candidates.extend(value)
            elif value is not None:
                candidates.append(value)

        for ann in raw_annotations or []:
            for key in ("ann_ids", "ann_id", "ann_id_list"):
                value = ann.get(key)
                if isinstance(value, (list, tuple)):
                    candidates.extend(value)
                elif value is not None:
                    candidates.append(value)

        seen = set()
        for cand in candidates:
            try:
                cid = int(cand)
            except (TypeError, ValueError):
                continue
            if cid not in seen:
                seen.add(cid)
                ann_ids.append(cid)
        return ann_ids

    @classmethod
    def _load_instance_data(cls, inst_json_path):
        if not inst_json_path:
            return None

        inst_json_path = os.path.abspath(inst_json_path)
        cache = cls._INSTANCE_DATA_CACHE
        if inst_json_path in cache:
            return cache[inst_json_path]

        try:
            with open(inst_json_path, "r") as f:
                inst_data = json.load(f)
        except OSError as exc:
            logger.error(
                "[RefCOCOMapper] Failed to load instances json %s: %s",
                inst_json_path,
                exc,
            )
            cache[inst_json_path] = {"annotations": {}, "image_hw": {}}
            return cache[inst_json_path]

        ann_map = {}
        for ann in inst_data.get("annotations", []):
            try:
                ann_id = int(ann["id"])
            except (KeyError, TypeError, ValueError):
                continue
            ann_map[ann_id] = ann

        image_hw = {}
        for img_meta in inst_data.get("images", []):
            try:
                img_id = int(img_meta["id"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                img_h = int(img_meta.get("height", 0) or 0)
                img_w = int(img_meta.get("width", 0) or 0)
            except (TypeError, ValueError):
                img_h, img_w = 0, 0
            image_hw[img_id] = (max(img_h, 1), max(img_w, 1))

        cache[inst_json_path] = {"annotations": ann_map, "image_hw": image_hw}
        return cache[inst_json_path]

    def _resolve_inst_json(self, dataset_dict):
        inst_json = dataset_dict.get("inst_json")
        if inst_json:
            return inst_json

        dataset_name = dataset_dict.get("dataset_name")
        if dataset_name:
            try:
                metadata = MetadataCatalog.get(dataset_name)
            except KeyError:
                metadata = None
            if metadata is not None:
                inst_json = getattr(metadata, "inst_json", None)
                if inst_json:
                    return inst_json

        inst_json = os.environ.get("MIAMI_INST_JSON")
        if inst_json:
            return inst_json

        return None

    @staticmethod
    def _find_raw_annotation(raw_annotations, ann_id):
        for ann in raw_annotations or []:
            values = []
            for key in ("ann_ids", "ann_id", "ann_id_list"):
                value = ann.get(key)
                if isinstance(value, (list, tuple)):
                    values.extend(value)
                elif value is not None:
                    values.append(value)
            if not values and ann.get("id") is not None:
                values.append(ann.get("id"))
            for value in values:
                try:
                    if int(value) == int(ann_id):
                        return ann
                except (TypeError, ValueError):
                    continue
        return None

    def _synthesize_mask(self, dataset_dict):
        fallback_shape = None
        image = dataset_dict.get("image")
        if torch.is_tensor(image):
            fallback_shape = (int(image.shape[-2]), int(image.shape[-1]))
        elif isinstance(image, np.ndarray):
            fallback_shape = image.shape[:2]

        height, width = self._normalize_hw(
            dataset_dict.get("height"),
            dataset_dict.get("width"),
            fallback_shape,
        )

        raw_annotations = dataset_dict.get("_raw_annotations") or []
        transformed_annotations = dataset_dict.get("annotations", []) or []

        ann_ids = self._collect_ann_ids(dataset_dict, raw_annotations)

        inst_json = self._resolve_inst_json(dataset_dict)
        if not inst_json and not self._warned_missing_inst_json:
            logger.warning(
                "[RefCOCOMapper] No instances json provided; relying on dataset annotations only."
            )
            self._warned_missing_inst_json = True
        inst_data = self._load_instance_data(inst_json) if inst_json else None

        ann_store = inst_data["annotations"] if inst_data else {}
        image_hw_store = inst_data["image_hw"] if inst_data else {}

        status_log = []
        per_instance_masks = []
        per_instance_statuses = []
        fallback_used = False

        def _accumulate_status(ann_identifier, status, mask_np):
            status_log.append(
                {
                    "ann_id": ann_identifier,
                    "status": status,
                    "mask_sum": int(mask_np.sum()) if mask_np is not None else 0,
                }
            )

        if ann_ids:
            for ann_id in ann_ids:
                ann_key = int(ann_id)
                ann_record = ann_store.get(ann_key) if ann_store else None
                if ann_record is None and self.id_to_ann:
                    ann_record = self.id_to_ann.get(ann_key)
                statuses = []
                mask_np = None
                mask_sum = 0

                original_hw = None
                if ann_record is not None:
                    image_id = ann_record.get("image_id")
                    if image_id is not None:
                        try:
                            original_hw = image_hw_store.get(int(image_id))
                        except (TypeError, ValueError):
                            original_hw = None

                    decoded_mask, decode_status = self._decode_annotation_mask(
                        ann_record, height, width, original_hw=original_hw
                    )
                    if decode_status:
                        statuses.append(decode_status)
                    if decoded_mask is not None and decoded_mask.sum() > 0:
                        mask_np = decoded_mask
                        mask_sum = int(decoded_mask.sum())

                if mask_np is None and ann_record is not None:
                    bbox_mode = ann_record.get("bbox_mode", "XYWH_ABS")
                    bbox_mask, bbox_status = bbox_to_mask(
                        ann_record.get("bbox"),
                        height,
                        width,
                        bbox_mode=bbox_mode,
                    )
                    if bbox_status:
                        statuses.append(bbox_status)
                    if bbox_mask is not None and bbox_mask.sum() > 0:
                        mask_np = bbox_mask
                        mask_sum = int(bbox_mask.sum())
                        fallback_used = True

                if mask_np is None:
                    raw_ann = self._find_raw_annotation(raw_annotations, ann_id)
                    if raw_ann is not None:
                        decoded_mask, decode_status = self._decode_annotation_mask(
                            raw_ann,
                            height,
                            width,
                            original_hw=fallback_shape,
                        )
                        if decode_status:
                            statuses.append(decode_status)
                        if decoded_mask is not None and decoded_mask.sum() > 0:
                            mask_np = decoded_mask
                            mask_sum = int(decoded_mask.sum())

                if mask_np is None and transformed_annotations:
                    for ann in transformed_annotations:
                        bbox_mask, bbox_status = bbox_to_mask(
                            ann.get("bbox"),
                            height,
                            width,
                            bbox_mode=ann.get("bbox_mode", BoxMode.XYXY_ABS),
                        )
                        if bbox_status:
                            statuses.append(bbox_status)
                        if bbox_mask is not None and bbox_mask.sum() > 0:
                            mask_np = bbox_mask
                            mask_sum = int(bbox_mask.sum())
                            fallback_used = True
                            break

                if mask_np is None:
                    synthetic_mask = np.zeros((height, width), dtype=np.uint8)
                    synthetic_mask[height // 2, width // 2] = 1
                    mask_np = synthetic_mask
                    mask_sum = 1
                    statuses.append("synthetic")
                    fallback_used = True

                status_label = "+".join([s for s in statuses if s]) or "missing"
                mask_np = np.ascontiguousarray((mask_np > 0).astype(np.uint8, copy=False))
                per_instance_masks.append(mask_np)
                per_instance_statuses.append(status_label)
                _accumulate_status(int(ann_id), status_label, mask_np)

        if not per_instance_masks and raw_annotations:
            for idx, raw_ann in enumerate(raw_annotations):
                decoded_mask, decode_status = self._decode_annotation_mask(
                    raw_ann,
                    height,
                    width,
                    original_hw=fallback_shape,
                )
                if decoded_mask is None or decoded_mask.sum() == 0:
                    continue
                mask_np = np.ascontiguousarray((decoded_mask > 0).astype(np.uint8, copy=False))
                per_instance_masks.append(mask_np)
                status = decode_status or "raw"
                per_instance_statuses.append(status)
                _accumulate_status(f"raw_{idx}", status, mask_np)

        final_mask, merged_status = merge_instance_masks(
            per_instance_masks,
            height,
            width,
            statuses=per_instance_statuses if per_instance_statuses else None,
        )

        if (final_mask is None or final_mask.sum() == 0) and transformed_annotations:
            bbox_masks = []
            bbox_statuses = []
            for ann in transformed_annotations:
                bbox_mask, bbox_status = bbox_to_mask(
                    ann.get("bbox"),
                    height,
                    width,
                    bbox_mode=ann.get("bbox_mode", BoxMode.XYXY_ABS),
                )
                if bbox_mask is not None and bbox_mask.sum() > 0:
                    bbox_masks.append(bbox_mask)
                bbox_statuses.append(bbox_status)

            if bbox_masks:
                final_mask, fallback_status = merge_instance_masks(
                    bbox_masks,
                    height,
                    width,
                    statuses=bbox_statuses,
                )
                fallback_used = True
                if final_mask is not None:
                    final_mask = np.ascontiguousarray((final_mask > 0).astype(np.uint8, copy=False))
                    _accumulate_status("bbox_fallback", fallback_status, final_mask)

        if final_mask is None or final_mask.sum() == 0:
            synthetic_mask = np.zeros((height, width), dtype=np.uint8)
            synthetic_mask[height // 2, width // 2] = 1
            final_mask = synthetic_mask
            fallback_used = True
            _accumulate_status("synthetic_fallback", "synthetic", final_mask)

        final_mask = np.ascontiguousarray((final_mask > 0).astype(np.uint8, copy=False))

        decode_counts = {"rle": 0, "poly": 0, "bbox": 0, "synthetic": 0}
        for entry in status_log:
            status_tokens = str(entry.get("status", "")).lower().split("+")
            tokens = {token.strip() for token in status_tokens if token.strip()}
            if any("rle" in token for token in tokens):
                decode_counts["rle"] += 1
            if any("poly" in token for token in tokens):
                decode_counts["poly"] += 1
            if any("bbox" in token for token in tokens):
                decode_counts["bbox"] += 1
            if any("synthetic" in token for token in tokens):
                decode_counts["synthetic"] += 1

        dataset_dict["height"], dataset_dict["width"] = height, width
        dataset_dict["mask_status"] = {
            "height": height,
            "width": width,
            "groups": status_log,
            "decode_counts": decode_counts,
            "fallback_used": fallback_used,
            "num_groups": len(per_instance_masks),
            "mask_sum": int(final_mask.sum()),
            "inst_json": inst_json,
        }
        return final_mask

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        raw_annos_input = dataset_dict.get("annotations", []) or []
        dataset_dict["_raw_annotations"] = copy.deepcopy(raw_annos_input)
        if "ann_ids" not in dataset_dict and "ann_id" in dataset_dict:
            dataset_dict["ann_ids"] = dataset_dict.get("ann_id")
        if "ann_id" not in dataset_dict and "ann_ids" in dataset_dict:
            dataset_dict["ann_id"] = dataset_dict.get("ann_ids")

        no_target_flag = bool(dataset_dict.get("no_target", False))
        dataset_dict["no_target"] = no_target_flag
        dataset_dict["empty"] = bool(dataset_dict.get("empty", no_target_flag))

        _src = dataset_dict.get("source", "miami2025")
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        dataset_dict["height"], dataset_dict["width"] = image_shape

        # USER: Implement additional transformations if you have other types of data
        annos = []
        for obj in dataset_dict.pop("annotations", []):
            if (obj.get("iscrowd", 0) != 0) or obj.get("empty", False):
                continue
            ann_identifier = obj.get("ann_ids") or obj.get("ann_id")
            if ("bbox" not in obj) or ("bbox_mode" not in obj):
                inferred = _infer_bbox_from_segmentation(obj)
                if inferred is not None:
                    obj["bbox"] = inferred
                    obj["bbox_mode"] = BoxMode.XYXY_ABS
                else:
                    continue
            transformed = utils.transform_instance_annotations(obj, transforms, image_shape)
            if ann_identifier is not None:
                transformed["ann_id"] = ann_identifier
                transformed["ann_ids"] = ann_identifier
            annos.append(transformed)
        dataset_dict["annotations"] = annos
        instances = utils.annotations_to_instances(annos, image_shape)

        empty = bool(dataset_dict.get("empty", False))

        if len(instances) > 0:
            if empty:
                logger.warning(
                    "[RefCOCOMapper] Sample image_id=%s marked as no_target but has annotations; treating as targeted.",
                    dataset_dict.get("image_id"),
                )
                empty = False
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Generate masks from polygon
            h, w = instances.image_size
            assert hasattr(instances, 'gt_masks')
            gt_masks = instances.gt_masks
            gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            instances.gt_masks = gt_masks
        else:
            if not empty:
                logger.warning(
                    "[RefCOCOMapper] Targeted sample image_id=%s produced no instances; using fallback mask only.",
                    dataset_dict.get("image_id"),
                )
            gt_masks = torch.zeros((0, image_shape[0], image_shape[1]), dtype=torch.uint8)
            instances.gt_masks = gt_masks

        merged_mask = self._synthesize_mask(dataset_dict)
        merged_mask = np.ascontiguousarray(merged_mask.astype(np.uint8, copy=False))
        dataset_dict.pop("_raw_annotations", None)
        if self.is_train:
            dataset_dict["gt_mask_merged_numpy"] = merged_mask
            dataset_dict["gt_mask_merged"] = torch.from_numpy(merged_mask.copy()).unsqueeze(0)
        else:
            dataset_dict["gt_mask_merged"] = merged_mask

        if self.is_train:
            dataset_dict["instances"] = instances
        else:
            dataset_dict["gt_mask"] = gt_masks

        dataset_dict["empty"] = empty

        # Language data
        dataset_name = dataset_dict.get("dataset_name", "") or ""
        sentence_field = dataset_dict.get("sentence", "")

        if "miami2025" in dataset_name:
            if isinstance(sentence_field, dict):
                sentence_raw = (
                    sentence_field.get("raw")
                    or sentence_field.get("sent")
                    or ""
                )
            else:
                sentence_raw = str(sentence_field)
        else:
            if isinstance(sentence_field, dict):
                sentence_raw = sentence_field.get("raw", "")
            else:
                sentence_raw = str(sentence_field)

        if not sentence_raw and dataset_dict.get("sentences"):
            first_sentence = dataset_dict["sentences"][0]
            if isinstance(first_sentence, dict):
                sentence_raw = (
                    first_sentence.get("raw")
                    or first_sentence.get("sent")
                    or ""
                )
            elif isinstance(first_sentence, str):
                sentence_raw = first_sentence

        attention_mask = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens

        input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

        input_ids = input_ids[:self.max_tokens]
        padded_input_ids[:len(input_ids)] = input_ids

        attention_mask[:len(input_ids)] = [1] * len(input_ids)

        dataset_dict['lang_tokens'] = torch.tensor(padded_input_ids).unsqueeze(0)
        dataset_dict['lang_mask'] = torch.tensor(attention_mask).unsqueeze(0)

        # ---- Preserve essential fields for downstream evaluator ------------
        dataset_dict["source"] = _src
        try:
            dataset_dict["ref_id"] = int(dataset_dict.get("ref_id", -1))
        except (TypeError, ValueError):
            dataset_dict["ref_id"] = -1
        if "sentence_info" not in dataset_dict:
            dataset_dict["sentence_info"] = sentence_field

        sent = sentence_raw
        if not isinstance(sent, str):
            sent = str(sent) if sent is not None else ""
        dataset_dict["sentence"] = sent

        return dataset_dict
    @staticmethod
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

