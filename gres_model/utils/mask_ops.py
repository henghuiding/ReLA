"""Robust mask decoding and synthesis utilities for Miami2025 and similar datasets."""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - optional dependency
    from pycocotools import mask as coco_mask
except ImportError:  # pragma: no cover
    coco_mask = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from detectron2.structures import BoxMode as _BoxMode
except ImportError:  # pragma: no cover
    _BoxMode = None  # type: ignore


LOGGER = logging.getLogger(__name__)

MaskArray = np.ndarray
SegmentationInput = Union[dict, List, Tuple]


def _resize_mask(mask: MaskArray, height: int, width: int) -> Optional[MaskArray]:
    """Resize ``mask`` to ``(height, width)`` with nearest-neighbour interpolation."""

    if mask is None:
        return None

    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.reshape(mask.shape[-2], mask.shape[-1])
    if mask.shape == (height, width):
        return np.ascontiguousarray(mask.astype(np.uint8, copy=False))

    if cv2 is not None:
        resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        return np.ascontiguousarray(resized.astype(np.uint8, copy=False))

    if torch is None or F is None:
        LOGGER.warning("[mask_ops] Falling back to numpy resize for mask shape %s", mask.shape)
        y_indices = (np.linspace(0, mask.shape[0] - 1, height)).round().astype(int)
        x_indices = (np.linspace(0, mask.shape[1] - 1, width)).round().astype(int)
        resized = mask[np.ix_(y_indices, x_indices)]
        return np.ascontiguousarray(resized.astype(np.uint8, copy=False))

    tensor = torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(height, width), mode="nearest")
    return resized.squeeze(0).squeeze(0).to(dtype=torch.uint8).cpu().numpy()


def _ensure_c_uint8(mask: Optional[MaskArray]) -> Optional[MaskArray]:
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim == 0:
        return None
    if mask.ndim > 2:
        mask = mask.reshape(mask.shape[-2], mask.shape[-1])
    return np.ascontiguousarray(mask.astype(np.uint8, copy=False))


def _convert_to_xyxy(bbox: Sequence[float], bbox_mode: Optional[Union[int, str]] = None) -> Optional[Tuple[float, float, float, float]]:
    """Convert ``bbox`` into XYXY_ABS format."""

    if bbox is None:
        return None

    if _BoxMode is not None:
        try:
            if bbox_mode is None:
                bbox_mode = _BoxMode.XYXY_ABS
            return tuple(_BoxMode.convert(bbox, bbox_mode, _BoxMode.XYXY_ABS))  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - fallback path
            LOGGER.debug("[mask_ops] BoxMode conversion failed; falling back to manual conversion.")

    # Manual conversion based on Detectron2 BoxMode constants.
    mode = bbox_mode
    if isinstance(mode, str):
        mode = mode.upper()
    if mode in (0, "XYXY_ABS", None):
        x0, y0, x1, y1 = bbox
    elif mode in (1, "XYWH_ABS"):
        x0, y0, w, h = bbox
        x1 = x0 + w
        y1 = y0 + h
    else:  # pragma: no cover - unexpected modes
        LOGGER.warning("[mask_ops] Unsupported bbox_mode %s; assuming XYXY_ABS", bbox_mode)
        x0, y0, x1, y1 = bbox
    return float(x0), float(y0), float(x1), float(y1)


def _sanitize_bbox(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """Clamp and ensure that the bbox covers at least one pixel."""

    width = max(int(width), 1)
    height = max(int(height), 1)

    x0i = int(np.floor(max(min(x0, width - 1), 0.0)))
    y0i = int(np.floor(max(min(y0, height - 1), 0.0)))
    x1i = int(np.ceil(max(min(x1, width), 0.0)))
    y1i = int(np.ceil(max(min(y1, height), 0.0)))

    if x1i <= x0i:
        x1i = min(width, max(x0i + 1, 1))
    if y1i <= y0i:
        y1i = min(height, max(y0i + 1, 1))

    if x0i >= width or y0i >= height or x1i <= 0 or y1i <= 0:
        return None

    x0i = max(x0i, 0)
    y0i = max(y0i, 0)
    x1i = min(x1i, width)
    y1i = min(y1i, height)

    if x1i <= x0i or y1i <= y0i:
        return None

    return x0i, y0i, x1i, y1i


def bbox_to_mask(
    bbox: Optional[Sequence[float]],
    height: int,
    width: int,
    *,
    bbox_mode: Optional[Union[int, str]] = None,
) -> Tuple[Optional[MaskArray], str]:
    """Create a binary mask from ``bbox`` with guaranteed non-zero area when possible."""

    if bbox is None:
        return None, "bbox_missing"

    xyxy = _convert_to_xyxy(bbox, bbox_mode=bbox_mode)
    if xyxy is None:
        return None, "bbox_invalid"

    x0, y0, x1, y1 = xyxy
    coords = _sanitize_bbox(x0, y0, x1, y1, width=width, height=height)
    if coords is None:
        return None, "bbox_invalid"

    x0i, y0i, x1i, y1i = coords
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0i:y1i, x0i:x1i] = 1
    if mask.sum() == 0:
        return None, "bbox_zero"
    return np.ascontiguousarray(mask), "bbox"


def _decode_rle(segmentation: SegmentationInput) -> Optional[MaskArray]:
    if coco_mask is None:
        raise RuntimeError("pycocotools is required for RLE decoding")
    return coco_mask.decode(segmentation)


def _decode_polygons(polygons: Sequence[Sequence[float]], height: int, width: int) -> Optional[MaskArray]:
    if coco_mask is None:
        raise RuntimeError("pycocotools is required for polygon decoding")
    if not polygons:
        return None
    rles = coco_mask.frPyObjects(polygons, height, width)
    return coco_mask.decode(rles)


def _collapse_mask(decoded: Optional[MaskArray]) -> Optional[MaskArray]:
    if decoded is None:
        return None
    decoded = np.asarray(decoded)
    if decoded.ndim == 3:
        decoded = decoded.max(axis=2)
    if decoded.size == 0:
        return None
    return decoded


def _safe_decode(
    decode_fn,
    *args,
    **kwargs,
) -> Tuple[Optional[MaskArray], Optional[str]]:
    try:
        mask = decode_fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - guarded failure path
        return None, f"error:{exc}"
    mask = _collapse_mask(mask)
    if mask is None or mask.sum() == 0:
        return None, "empty"
    return mask, None


def _poly_to_mask_safe(
    polygons: Sequence[Sequence[float]],
    height: int,
    width: int,
    *,
    original_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[MaskArray], str]:
    """Decode polygon segmentations safely and resize to ``(height, width)``."""

    if not polygons:
        return None, "poly_missing"

    base_h, base_w = original_size if original_size else (height, width)
    base_h = max(int(base_h), 1)
    base_w = max(int(base_w), 1)
    mask, status = _safe_decode(_decode_polygons, polygons, base_h, base_w)
    if mask is None:
        return None, f"poly_{status or 'invalid'}"

    mask = _resize_mask(mask, height, width)
    if mask is None or mask.sum() == 0:
        return None, "poly_empty"
    return mask, "poly"


def _rle_to_mask_safe(
    segmentation: SegmentationInput,
    height: int,
    width: int,
    *,
    original_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[MaskArray], str]:
    """Decode RLE segmentations safely and resize to ``(height, width)``."""

    if not segmentation:
        return None, "rle_missing"

    base_h = base_w = None
    if original_size:
        base_h = max(int(original_size[0]), 1)
        base_w = max(int(original_size[1]), 1)

    seg_obj = segmentation
    if isinstance(segmentation, dict):
        seg_obj = dict(segmentation)
        if base_h is not None and base_w is not None and not seg_obj.get("size"):
            seg_obj["size"] = [base_h, base_w]
    elif isinstance(segmentation, (list, tuple)) and segmentation and isinstance(segmentation[0], dict):
        seg_list = []
        for piece in segmentation:
            piece_dict = dict(piece)
            if base_h is not None and base_w is not None and not piece_dict.get("size"):
                piece_dict["size"] = [base_h, base_w]
            seg_list.append(piece_dict)
        seg_obj = seg_list

    try:
        decoded, status = _safe_decode(_decode_rle, seg_obj)
    except RuntimeError as exc:  # pragma: no cover - pycocotools missing
        LOGGER.error("[mask_ops] %s", exc)
        return None, "rle_unavailable"

    if decoded is None:
        return None, f"rle_{status or 'invalid'}"

    mask = _resize_mask(decoded, height, width)
    if mask is None or mask.sum() == 0:
        return None, "rle_empty"
    return mask, "rle"


def merge_instance_masks(
    masks: Iterable[Optional[MaskArray]],
    height: int,
    width: int,
    *,
    statuses: Optional[Sequence[Optional[str]]] = None,
) -> Tuple[Optional[MaskArray], str]:
    """Merge ``masks`` via logical OR and return a contiguous uint8 array."""

    merged = np.zeros((height, width), dtype=np.uint8)
    valid = False

    for mask in masks:
        mask_np = _ensure_c_uint8(mask)
        if mask_np is None:
            continue
        if mask_np.shape != (height, width):
            mask_np = _resize_mask(mask_np, height, width)
            if mask_np is None:
                continue
        merged = np.maximum(merged, (mask_np > 0).astype(np.uint8))
        valid = True

    if not valid or merged.sum() == 0:
        status = "empty"
    else:
        status = "merged"

    if statuses:
        non_empty = [s for s in statuses if s]
        if non_empty:
            status = "+".join(sorted(set(non_empty)))
            if valid and merged.sum() > 0 and "bbox" not in status and "poly" not in status and "rle" not in status:
                status = f"merged({status})"

    if not valid or merged.sum() == 0:
        return None, status
    return np.ascontiguousarray(merged.astype(np.uint8, copy=False)), status


__all__ = [
    "_rle_to_mask_safe",
    "_poly_to_mask_safe",
    "bbox_to_mask",
    "merge_instance_masks",
]
