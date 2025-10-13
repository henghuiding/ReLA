import math
import time
import copy
import json
import logging
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    _banner_printed = False

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens=None,
        image_format="RGB",
        bert_type="bert-base-uncased",
        max_tokens=32,
        merge=True,
        preload_only=False,
    ):
        self.is_train = is_train
        self.merge = merge
        self.preload_only = preload_only
        self.tfm_gens = tfm_gens if tfm_gens is not None else []
        if not self.preload_only:
            logging.getLogger(__name__).info(
                "Full TransformGens used: {}".format(str(self.tfm_gens))
            )

        self.bert_type = bert_type
        self.max_tokens = max_tokens
        if not self.preload_only:
            logging.getLogger(__name__).info(
                "Loading BERT tokenizer: {}...".format(self.bert_type)
            )
            self.tokenizer = BertTokenizer.from_pretrained("/autodl-tmp/bert-base-uncased")

        else:
            self.tokenizer = None

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
                if not getattr(RefCOCOMapper, "_banner_printed", False):
                    print(
                        f"[RefCOCOMapper] Preloaded {len(self.id_to_ann)} instance annotations from {default_inst_path}"
                    )
                    RefCOCOMapper._banner_printed = True
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
            "preload_only": False,
        }
        return ret

    @staticmethod
    def _merge_masks(x):
        return x.sum(dim=0, keepdim=True).clamp(max=1)

    @staticmethod
    def _parse_hw(value: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) < 2:
                return None
            cand_h, cand_w = value[0], value[1]
        else:
            return None
        try:
            if isinstance(cand_h, np.ndarray):
                cand_h = cand_h.item()
            if isinstance(cand_w, np.ndarray):
                cand_w = cand_w.item()
            cand_h = int(round(float(cand_h)))
            cand_w = int(round(float(cand_w)))
        except (TypeError, ValueError):
            return None
        if cand_h <= 1 or cand_w <= 1:
            return None
        return cand_h, cand_w

    @staticmethod
    def _infer_canvas_hw_from_segmentation(segmentation) -> Optional[Tuple[int, int]]:
        if not segmentation:
            return None

        def _sanitize_from_size(size_obj) -> Optional[Tuple[int, int]]:
            if not isinstance(size_obj, (list, tuple)) or len(size_obj) < 2:
                return None
            try:
                sh = int(round(float(size_obj[0])))
                sw = int(round(float(size_obj[1])))
            except (TypeError, ValueError):
                return None
            if sh <= 1 or sw <= 1:
                return None
            return sh, sw

        if isinstance(segmentation, dict):
            size = segmentation.get("size")
            size_dims = _sanitize_from_size(size)
            if size_dims:
                return size_dims
            return None

        max_x = 0.0
        max_y = 0.0
        found_poly = False
        if isinstance(segmentation, (list, tuple)):
            for piece in segmentation:
                if isinstance(piece, dict):
                    dims = RefCOCOMapper._infer_canvas_hw_from_segmentation(piece)
                    if dims:
                        return dims
                    continue
                if not isinstance(piece, (list, tuple)):
                    continue
                coords: List[float] = []
                for coord in piece:
                    try:
                        coords.append(float(coord))
                    except (TypeError, ValueError):
                        continue
                if len(coords) < 2:
                    continue
                xs = coords[0::2]
                ys = coords[1::2]
                if xs:
                    max_x = max(max_x, max(xs))
                if ys:
                    max_y = max(max_y, max(ys))
                found_poly = True

        if not found_poly:
            return None

        height = int(math.ceil(max_y + 1.0))
        width = int(math.ceil(max_x + 1.0))
        if height <= 1 or width <= 1:
            return None
        return height, width

    @staticmethod
    def _infer_canvas_hw_from_annotation(ann: Dict[str, object]) -> Optional[Tuple[int, int]]:
        if not isinstance(ann, dict):
            return None

        seg_dims = RefCOCOMapper._infer_canvas_hw_from_segmentation(ann.get("segmentation"))
        if seg_dims:
            return seg_dims

        bbox = ann.get("bbox")
        if bbox is not None:
            bbox_mode = ann.get("bbox_mode", BoxMode.XYWH_ABS)
            try:
                x0, y0, x1, y1 = BoxMode.convert(bbox, bbox_mode, BoxMode.XYXY_ABS)
                width = int(math.ceil(max(x1, 0.0)))
                height = int(math.ceil(max(y1, 0.0)))
                if height > 1 and width > 1:
                    return height, width
            except Exception:
                logger.debug(
                    "[RefCOCOMapper] Failed to infer canvas from bbox for ann_id=%s",
                    ann.get("id"),
                )
        return None

    @classmethod
    def _normalize_hw(
        cls,
        sources: Sequence[Tuple[str, Optional[Sequence[int]]]],
        *,
        dataset_dict: Optional[Dict[str, object]] = None,
    ) -> Tuple[int, int]:
        """Resolve a unique ``(height, width)`` pair from trusted sources."""

        valid_sources: List[Tuple[str, Tuple[int, int]]] = []
        for name, candidate in sources:
            parsed = cls._parse_hw(candidate)
            if parsed is not None:
                valid_sources.append((name, parsed))

        if not valid_sources:
            raise AssertionError(
                "[RefCOCOMapper] No valid height/width sources provided; cannot proceed"
            )

        trusted_names = {
            name
            for name, _ in valid_sources
            if name in {"image_tensor", "instances_json", "poly_bounds"}
        }
        if not trusted_names:
            raise AssertionError(
                "[RefCOCOMapper] Missing trusted height/width source (image tensor or instances.json)"
            )

        priority = {
            "image_tensor": 0,
            "instances_json": 1,
            "poly_bounds": 2,
            "dataset_fields": 3,
            "original_hw": 4,
        }

        candidates: List[Tuple[bool, int, int, int, str, Tuple[int, int]]] = []
        for idx, (name, dims) in enumerate(valid_sources):
            h, w = dims
            area = max(h, 0) * max(w, 0)
            non_trivial = h > 1 and w > 1
            trust_rank = priority.get(name, 10)
            candidates.append((non_trivial, area, -trust_rank, -idx, name, dims))

        unique_dims = {entry[-1] for entry in candidates}
        if len(unique_dims) > 1:
            logger.warning(
                f"[RefCOCOMapper] Warning: Conflicting H/W sources {valid_sources}, auto-resolving..."
            )

        non_trivial_candidates = [c for c in candidates if c[0]]
        search_space = non_trivial_candidates or candidates
        best = max(search_space, key=lambda item: (item[1], item[2], item[3]))
        height, width = best[-1]

        if height <= 1 or width <= 1:
            # Final safeguard: pick the largest area even if degenerate.
            fallback = max(candidates, key=lambda item: (item[1], item[2], item[3]))
            height, width = fallback[-1]

        if dataset_dict is not None:
            assertions = dataset_dict.setdefault("_mask_assertions", {})
            assertions["has_hw_from_images_json"] = (
                "instances_json" in trusted_names
                or "image_tensor" in trusted_names
                or "poly_bounds" in trusted_names
            )
            assertions["no_1x1_anywhere"] = height > 1 and width > 1

        assert height > 1 and width > 1, "[RefCOCOMapper] Invalid normalized shape"
        return height, width

    @staticmethod
    def _status_to_mode(status):
        tokens = str(status or "").lower().replace("(", "+").replace(")", "").split("+")
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if "poly" in token:
                return "POLY"
            if "rle" in token:
                return "RLE"
            if "bbox" in token:
                return "BBOX"
            if "synthetic" in token:
                return "SYNTHETIC"
        return "UNKNOWN"

    @staticmethod
    def _decode_annotation_mask(
        ann: Dict[str, object],
        height: int,
        width: int,
        *,
        original_hw: Optional[Sequence[int]] = None,
        dataset_dict: Optional[Dict[str, object]] = None,
    ):
        assert height > 1 and width > 1, "decode_inputs_have_hw"
        if dataset_dict is not None:
            gates = dataset_dict.setdefault("_mask_assertions", {})
            gates.setdefault("polygon_even_len", True)
            gates.setdefault("rle_counts_is_str", True)
            gates["decode_inputs_have_hw"] = True
        seg = ann.get("segmentation")
        masks = []
        statuses = []
        original_size = None
        if original_hw:
            try:
                oh, ow = int(original_hw[0]), int(original_hw[1])
                if oh > 1 and ow > 1:
                    original_size = (oh, ow)
            except (TypeError, ValueError):
                original_size = None

        def _mark_gate(name: str, ok: bool):
            if dataset_dict is None:
                return
            dataset_dict.setdefault("_mask_assertions", {})[name] = bool(ok)

        if isinstance(seg, dict):
            counts = seg.get("counts") if isinstance(seg, dict) else None
            if counts is not None:
                _mark_gate("rle_counts_is_str", isinstance(counts, (str, bytes)))
            mask, status = _rle_to_mask_safe(seg, height, width, original_size=original_size)
            if mask is not None:
                masks.append(mask)
            statuses.append(status)
        elif isinstance(seg, (list, tuple)):
            if not seg:
                statuses.append("seg_missing")
                raise ValueError("Empty segmentation list encountered")
            elif all(isinstance(poly, (list, tuple)) for poly in seg):
                for poly in seg:
                    if poly and len(poly) % 2 != 0:
                        _mark_gate("polygon_even_len", False)
                        raise AssertionError("polygon_even_len")
                _mark_gate("polygon_even_len", True)
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
                        if piece and len(piece) % 2 != 0:
                            _mark_gate("polygon_even_len", False)
                            raise AssertionError("polygon_even_len")
                        piece_mask, piece_status = _poly_to_mask_safe(
                            [piece], height, width, original_size=original_size
                        )
                    else:
                        raise ValueError("segmentation format unsupported")
                    if piece_mask is not None:
                        masks.append(piece_mask)
                    statuses.append(piece_status)
        else:
            raise ValueError("Unsupported segmentation type")

        merged_mask, status = merge_instance_masks(masks, height, width, statuses=statuses)
        if dataset_dict is not None and merged_mask is not None:
            dataset_dict.setdefault("_mask_assertions", {})["decode_outputs_area_gt0"] = (
                int(merged_mask.sum()) > 0
            )
        if merged_mask is None or merged_mask.sum() == 0:
            raise ValueError("Decoded mask is empty")
        return merged_mask, status

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
            if cid < 0:
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

        assertions = dataset_dict.setdefault("_mask_assertions", {})
        assertions.setdefault("coords_contract_ok", True)
        assertions.setdefault("bbox_fallback_uses_true_hw", True)

        raw_annotations = dataset_dict.get("_raw_annotations") or []
        ann_ids_seed = self._collect_ann_ids(dataset_dict, raw_annotations)

        ann_store_lookup = self.id_to_ann or {}

        image = dataset_dict.get("image")
        tensor_hw: Optional[Tuple[int, int]] = None
        if torch.is_tensor(image):
            tensor_hw = (int(image.shape[-2]), int(image.shape[-1]))
        elif isinstance(image, np.ndarray):
            tensor_hw = (int(image.shape[0]), int(image.shape[1]))

        stored_hw: Optional[Tuple[int, int]] = None
        try:
            stored_h = dataset_dict.get("height")
            stored_w = dataset_dict.get("width")
            if stored_h is not None and stored_w is not None:
                stored_hw = (int(stored_h), int(stored_w))
        except (TypeError, ValueError):
            stored_hw = None

        original_hw = dataset_dict.get("_original_hw")

        inst_json = self._resolve_inst_json(dataset_dict)
        if not inst_json and not self._warned_missing_inst_json:
            logger.warning(
                "[RefCOCOMapper] No instances json provided; relying on dataset annotations only."
            )
            self._warned_missing_inst_json = True
        inst_data = self._load_instance_data(inst_json) if inst_json else None

        ann_store = inst_data["annotations"] if inst_data else {}
        image_hw_store = inst_data["image_hw"] if inst_data else {}

        poly_hw: Optional[Tuple[int, int]] = None
        if ann_ids_seed:
            for ann_id in ann_ids_seed:
                try:
                    ann_key = int(ann_id)
                except (TypeError, ValueError):
                    continue
                ann_record = ann_store_lookup.get(ann_key) or ann_store.get(ann_key)
                if not ann_record:
                    continue
                dims = self._infer_canvas_hw_from_annotation(ann_record)
                if dims is None:
                    continue
                if poly_hw is None:
                    poly_hw = dims
                else:
                    poly_hw = (max(poly_hw[0], dims[0]), max(poly_hw[1], dims[1]))

        if poly_hw is None and raw_annotations:
            for ann in raw_annotations:
                dims = self._infer_canvas_hw_from_annotation(ann)
                if dims is None:
                    continue
                if poly_hw is None:
                    poly_hw = dims
                else:
                    poly_hw = (max(poly_hw[0], dims[0]), max(poly_hw[1], dims[1]))

        image_meta_hw: Optional[Tuple[int, int]] = None
        image_id = dataset_dict.get("image_id")
        if image_id is not None:
            try:
                image_meta_hw = image_hw_store.get(int(image_id))
            except (TypeError, ValueError):
                image_meta_hw = None

        if dataset_dict.get("image") is None and image_meta_hw is not None:
            dataset_dict["height"], dataset_dict["width"] = image_meta_hw
            stored_hw = image_meta_hw
            assertions["offline_mode_has_hw"] = True
        elif dataset_dict.get("image") is None:
            assertions["offline_mode_has_hw"] = False

        height, width = self._normalize_hw(
            [
                ("image_tensor", tensor_hw),
                ("dataset_fields", stored_hw),
                ("original_hw", original_hw),
                ("instances_json", image_meta_hw),
                ("poly_bounds", poly_hw),
            ],
            dataset_dict=dataset_dict,
        )

        transformed_annotations = dataset_dict.get("annotations", []) or []

        ann_ids_raw = list(ann_ids_seed)
        valid_ann_ids: List[int] = []
        for ann in ann_ids_raw:
            try:
                ann_int = int(ann)
            except (TypeError, ValueError):
                continue
            if ann_int < 0:
                continue
            valid_ann_ids.append(ann_int)

        merged_mask = np.zeros((height, width), dtype=np.uint8)
        status_log: List[Dict[str, object]] = []
        missing_ann_ids: List[int] = []
        had_bbox_fallback = False
        had_poly_empty = False
        had_exception = False
        contributing = 0

        for ann_id in valid_ann_ids:
            ann_key = int(ann_id)
            ann_record = ann_store_lookup.get(ann_key) or ann_store.get(ann_key)
            current_status: List[str] = []
            mask_np: Optional[np.ndarray] = None
            decode_error: Optional[str] = None

            original_candidate = dataset_dict.get("_original_hw")
            if ann_record is not None:
                ann_image_id = ann_record.get("image_id")
                if ann_image_id is not None:
                    try:
                        mapped = image_hw_store.get(int(ann_image_id))
                    except (TypeError, ValueError):
                        mapped = None
                    if mapped:
                        original_candidate = mapped

            if ann_record is not None:
                try:
                    mask_np, decode_status = self._decode_annotation_mask(
                        ann_record,
                        height,
                        width,
                        original_hw=original_candidate,
                        dataset_dict=dataset_dict,
                    )
                    current_status.append(decode_status)
                except ValueError as exc:
                    decode_error = str(exc)
                    had_exception = True

            if mask_np is None:
                if ann_record is None:
                    missing_ann_ids.append(ann_key)
                if ann_record is not None and ann_record.get("bbox") is not None:
                    bbox_mask, bbox_status = bbox_to_mask(
                        ann_record.get("bbox"),
                        height,
                        width,
                        bbox_mode=ann_record.get("bbox_mode", BoxMode.XYWH_ABS),
                    )
                    if bbox_mask is not None and bbox_mask.sum() > 0:
                        if bbox_mask.shape != (height, width):
                            raise AssertionError("bbox_fallback_uses_true_hw")
                        mask_np = bbox_mask
                        current_status.append(bbox_status)
                        had_bbox_fallback = True
                        assertions["bbox_fallback_uses_true_hw"] = True
                elif decode_error is not None:
                    logger.error(
                        "[RefCOCOMapper] Mask decode failed for ann_id=%s: %s",
                        ann_key,
                        decode_error,
                    )
                    raise

            

            if mask_np is None and transformed_annotations:
                for ann in transformed_annotations:
                    bbox_mask, bbox_status = bbox_to_mask(
                        ann.get("bbox"),
                        height,
                        width,
                        bbox_mode=ann.get("bbox_mode", BoxMode.XYXY_ABS),
                    )
                    if bbox_mask is not None and bbox_mask.sum() > 0:
                        if bbox_mask.shape != (height, width):
                            raise AssertionError("bbox_fallback_uses_true_hw")
                        mask_np = bbox_mask
                        current_status.append(bbox_status)
                        had_bbox_fallback = True
                        assertions["bbox_fallback_uses_true_hw"] = True
                        break

            if mask_np is None:
                logger.error(
                    "[RefCOCOMapper] Unable to obtain mask for ann_id=%s after strict policy",
                    ann_key,
                )
                raise AssertionError("Strict rejection policy triggered")

            mask_np = np.ascontiguousarray(mask_np.astype(np.uint8, copy=False))
            if mask_np.shape != (height, width):
                raise AssertionError(
                    f"Decoded mask shape {mask_np.shape} does not match expected {(height, width)}"
                )

            mask_area = int(mask_np.sum())
            if mask_area == 0:
                had_poly_empty = had_poly_empty or any("poly" in s for s in current_status)
                logger.error(
                    "[RefCOCOMapper] Zero-area mask for ann_id=%s violates area contract",
                    ann_key,
                )
                raise AssertionError("decode_outputs_area_gt0")

            merged_mask = np.maximum(merged_mask, mask_np)
            contributing += 1

            if ann_record is not None and ann_record.get("bbox") is not None:
                try:
                    bbox_mode = ann_record.get("bbox_mode", BoxMode.XYWH_ABS)
                    xyxy = BoxMode.convert(ann_record.get("bbox"), bbox_mode, BoxMode.XYXY_ABS)

                    ys, xs = np.nonzero(mask_np)
                    mask_x0 = int(xs.min()) if xs.size else 0
                    mask_y0 = int(ys.min()) if ys.size else 0
                    mask_x1 = int(xs.max() + 1) if xs.size else 0
                    mask_y1 = int(ys.max() + 1) if ys.size else 0

                    ref_hw = original_candidate or dataset_dict.get("_original_hw")
                    if ref_hw is None:
                        ref_hw = (height, width)

                    try:
                        oh = max(int(ref_hw[0]), 1)
                        ow = max(int(ref_hw[1]), 1)
                    except (TypeError, ValueError):
                        oh, ow = height, width

                    if oh <= 0 or ow <= 0:
                        oh, ow = height, width

                    sx = float(width) / float(ow)
                    sy = float(height) / float(oh)

                    bx0 = xyxy[0] * sx
                    by0 = xyxy[1] * sy
                    bx1 = xyxy[2] * sx
                    by1 = xyxy[3] * sy

                    tol = max(1.5, 0.005 * float(max(height, width)))

                    coords_ok = (
                        mask_x0 >= math.floor(bx0 - tol)
                        and mask_y0 >= math.floor(by0 - tol)
                        and mask_x1 <= math.ceil(bx1 + tol)
                        and mask_y1 <= math.ceil(by1 + tol)
                    )

                    is_bbox_used = any(("bbox" in (s or "").lower()) for s in current_status)

                    assertions["coords_contract_ok"] = bool(coords_ok)
                    if not coords_ok:
                        msg = (
                            f"[RefCOCOMapper] coords_contract mismatch (scaled bbox "
                            f"({bx0:.1f},{by0:.1f},{bx1:.1f},{by1:.1f}) vs "
                            f"mask_box ({mask_x0},{mask_y0},{mask_x1},{mask_y1})) "
                            f"@canvas ({height},{width}), tol={tol:.2f}, ann_id={ann_key}"
                        )
                        if is_bbox_used:
                            logger.error(msg)
                            raise AssertionError("coords_contract_ok")
                        logger.warning(msg)
                except Exception as exc:
                    logger.error(
                        "[RefCOCOMapper] Coordinate contract failure for ann_id=%s: %s",
                        ann_key,
                        exc,
                    )
                    raise

            status_log.append(
                {
                    "ann_id": ann_key,
                    "status": "+".join(current_status) if current_status else "unknown",
                    "mask_sum": mask_area,
                    "error": decode_error,
                }
            )
        if contributing == 0 and not bool(dataset_dict.get("no_target", False)):
            raise AssertionError("merge_outputs_area_gt0")

        total_area = int(merged_mask.sum())
        assertions["merge_outputs_area_gt0"] = total_area > 0 or bool(dataset_dict.get("no_target", False))
        assertions["decode_outputs_area_gt0"] = total_area > 0 or bool(dataset_dict.get("no_target", False))

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

        if decode_counts["synthetic"] and not (
            decode_counts["rle"] or decode_counts["poly"] or decode_counts["bbox"]
        ):
            mode_label = "SYNTHETIC"
        elif decode_counts["rle"] + decode_counts["poly"] > 1:
            mode_label = "MERGED"
        elif decode_counts["rle"] + decode_counts["poly"] == 1:
            mode_label = "MASK"
        elif decode_counts["bbox"] > 0:
            mode_label = "BBOX"
        else:
            mode_label = "UNKNOWN"

        nonzero_ratio = mask_area / float(height * width) if height and width else 0.0
        merge_ratio = contributing / float(max(len(valid_ann_ids), 1)) if valid_ann_ids else 0.0
        assertions["merge_outputs_ratio"] = merge_ratio

        dataset_dict["mask_status"] = {
            "height": height,
            "width": width,
            "original_hw": dataset_dict.get("_original_hw"),
            "groups": status_log,
            "decode_counts": decode_counts,
            "fallback_used": had_bbox_fallback,
            "num_groups": len(status_log),
            "mask_sum": total_area,
            "inst_json": inst_json,
            "mode": mode_label,
            "missing_ann_ids": missing_ann_ids,
            "used_synthetic": decode_counts["synthetic"] > 0,
            "contributing": contributing,
            "merge_ratio": merge_ratio,
            "nonzero_ratio": nonzero_ratio,
        }

        mode_used = mode_label
        mask_area = total_area
        mode_source = "OFFLINE" if dataset_dict.get("image") is None else "ONLINE"
        ann_ids_csv = ";".join(str(a) for a in valid_ann_ids) if valid_ann_ids else ""
        csv_line = (
            f"{dataset_dict.get('image_id','unknown')},{ann_ids_csv},{mode_used},{mode_source},{height},{width},"
            f"{mask_area},{contributing},{int(had_bbox_fallback)},{int(had_poly_empty)},{int(had_exception)}"
        )
        logger.info("[RefCOCOMapper][audit]%s", csv_line)
        dataset_dict["mask_audit_csv"] = csv_line

        image_id = dataset_dict.get("image_id", "unknown")
        ann_summary = dataset_dict.get("ann_ids") or dataset_dict.get("ann_id") or []
        if not isinstance(ann_summary, (list, tuple)):
            ann_summary = [ann_summary]
        normalized_ann_summary: List[int] = []
        for value in ann_summary:
            try:
                normalized_ann_summary.append(int(value))
            except (TypeError, ValueError):
                continue

        source_desc = inst_json or ("preloaded-cache" if self.id_to_ann else "annotations-only")
        logger.info(
            "[RefCOCOMapper] sample %s | ann_ids %s | mode: %s | mask.shape %s | sum=%d | fallback=%s | source=%s",
            image_id,
            normalized_ann_summary,
            mode_label,
            tuple(merged_mask.shape),
            mask_area,
            had_bbox_fallback,
            source_desc,
        )
        logger.debug(
            "[RefCOCOMapper] final mode=%s | mask.shape=%s | mask.sum=%d",
            mode_label,
            tuple(merged_mask.shape),
            mask_area,
        )
        return merged_mask

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        dataset_dict["_mask_assertions"] = {}
        original_hw = None
        try:
            orig_h = int(dataset_dict.get("height"))
            orig_w = int(dataset_dict.get("width"))
            if orig_h > 0 and orig_w > 0:
                original_hw = (orig_h, orig_w)
        except (TypeError, ValueError):
            original_hw = None
        dataset_dict["_original_hw"] = original_hw
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
        dataset_dict.setdefault("_mask_assertions", {})["post_aug_shape_contract"] = (
            dataset_dict["image"].shape[-2:] == tuple(image_shape)
        )

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

        dataset_dict.pop("_original_hw", None)
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


def _run_internal_tests():
    """Lightweight self-checks to ensure mapper helpers behave as expected."""

    mapper = RefCOCOMapper(is_train=False, tfm_gens=[], preload_only=True)

    # Sample 1: Conflicting height/width sources resolved to the largest canvas.
    dataset_a = {
        "image": torch.zeros((3, 48, 48), dtype=torch.float32),
        "height": 48,
        "width": 48,
        "ann_ids": [1],
        "annotations": [],
        "_raw_annotations": [],
        "_original_hw": (96, 48),
        "image_id": 101,
        "no_target": False,
        "_mask_assertions": {},
    }
    mapper.id_to_ann = {
        1: {
            "id": 1,
            "bbox": [20, 20, 16, 16],
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": [[20, 20, 36, 20, 36, 36, 20, 36]],
        }
    }
    mask_a = mapper._synthesize_mask(dataset_a)
    print(
        f"[RefCOCOMapper][test] sample-A | hw={mask_a.shape} | "
        f"coords_contract_ok={dataset_a.get('_mask_assertions', {}).get('coords_contract_ok')}"
    )

    # Sample 2: Empty segmentation triggers bbox fallback while respecting contracts.
    dataset_b = {
        "image": torch.zeros((3, 64, 64), dtype=torch.float32),
        "height": 64,
        "width": 64,
        "ann_ids": [2],
        "annotations": [],
        "_raw_annotations": [],
        "_original_hw": (64, 64),
        "image_id": 202,
        "no_target": False,
        "_mask_assertions": {},
    }
    mapper.id_to_ann = {
        2: {
            "id": 2,
            "bbox": [16, 16, 20, 20],
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": [],
        }
    }
    mask_b = mapper._synthesize_mask(dataset_b)
    print(
        f"[RefCOCOMapper][test] sample-B | hw={mask_b.shape} | "
        f"coords_contract_ok={dataset_b.get('_mask_assertions', {}).get('coords_contract_ok')}"
    )


if __name__ == "__main__":
    _run_internal_tests()
