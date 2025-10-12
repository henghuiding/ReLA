"""Minimal smoke test for the Miami2025 evaluator pipeline."""

import argparse
import importlib.util
import itertools
import json
import os
import sys
from collections import Counter
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from gres_model.utils.mask_ops import (
        _poly_to_mask_safe,
        _rle_to_mask_safe,
        bbox_to_mask,
        merge_instance_masks,
    )
    from gres_model.data.dataset_mappers.refcoco_mapper import RefCOCOMapper
except ModuleNotFoundError:
    mask_ops_path = os.path.join(REPO_ROOT, "gres_model", "utils", "mask_ops.py")
    spec = importlib.util.spec_from_file_location("mask_ops_fallback", mask_ops_path)
    if spec is None or spec.loader is None:
        raise
    mask_ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mask_ops)  # type: ignore[attr-defined]
    _poly_to_mask_safe = mask_ops._poly_to_mask_safe
    _rle_to_mask_safe = mask_ops._rle_to_mask_safe
    bbox_to_mask = mask_ops.bbox_to_mask
    merge_instance_masks = mask_ops.merge_instance_masks
    RefCOCOMapper = None  # type: ignore[assignment]


_MAPPER_CACHE: Optional["RefCOCOMapper"] = None
_MAPPER_SOURCE: Optional[str] = None


def _load_default_instances(default_inst_path: str) -> Optional[SimpleNamespace]:
    if not os.path.exists(default_inst_path):
        print(
            f"[smoke_eval] WARNING: default instances file not found at {default_inst_path}"
        )
        return None

    try:
        with open(default_inst_path, "r", encoding="utf-8") as f:
            inst_data = json.load(f)
    except (OSError, ValueError, TypeError) as exc:
        print(
            "[smoke_eval] WARNING: Failed to preload default instances json "
            f"{default_inst_path}: {exc}"
        )
        return None

    anns = inst_data.get("annotations", []) or []
    id_to_ann = {
        int(ann["id"]): ann
        for ann in anns
        if isinstance(ann, dict) and "id" in ann
    }
    print(
        f"[smoke_eval] Preloaded {len(id_to_ann)} instance annotations from {default_inst_path}"
    )
    return SimpleNamespace(id_to_ann=id_to_ann)


def _ensure_mapper_preloaded() -> Optional["RefCOCOMapper"]:
    global _MAPPER_CACHE, _MAPPER_SOURCE
    if _MAPPER_CACHE is not None:
        return _MAPPER_CACHE

    mapper_cls = globals().get("RefCOCOMapper")
    if mapper_cls is not None:
        try:
            print(
                "[smoke_eval] Initializing RefCOCOMapper to preload instance annotations..."
            )
            _MAPPER_CACHE = mapper_cls(
                is_train=False,
                tfm_gens=[],
                image_format="RGB",
                bert_type="bert-base-uncased",
                max_tokens=32,
                merge=True,
                preload_only=True,
            )
            _MAPPER_SOURCE = "RefCOCOMapper"
            return _MAPPER_CACHE
        except Exception as exc:  # pragma: no cover - diagnostic output only
            print(
                "[smoke_eval] WARNING: Failed to instantiate RefCOCOMapper for preload: "
                f"{exc}"
            )

    default_inst_path = "/autodl-tmp/rela_data/annotations/instances.json"
    namespace_cache = _load_default_instances(default_inst_path)
    if namespace_cache is not None:
        _MAPPER_CACHE = namespace_cache
        _MAPPER_SOURCE = default_inst_path
    else:
        _MAPPER_CACHE = SimpleNamespace(id_to_ann={})
        _MAPPER_SOURCE = "missing"
    return _MAPPER_CACHE


def _describe_mapper_source() -> str:
    if _MAPPER_CACHE is None:
        return "uninitialized"
    if isinstance(_MAPPER_CACHE, SimpleNamespace):
        return _MAPPER_SOURCE or "json-cache"
    return _MAPPER_SOURCE or "RefCOCOMapper"


def _as_numpy_mask(mask, height: int, width: int) -> np.ndarray:
    """Return a numpy uint8 mask with shape ``(height, width)``."""

    if mask is None:
        return np.zeros((height, width), dtype=np.uint8)

    if isinstance(mask, np.ndarray):
        mask_np = mask.astype(np.uint8, copy=False)
    elif torch.is_tensor(mask):
        mask_cpu = mask.detach().cpu()
        if mask_cpu.dtype != torch.uint8:
            mask_cpu = mask_cpu.to(dtype=torch.uint8)
        mask_np = mask_cpu.numpy()
    else:
        mask_np = np.asarray(mask, dtype=np.uint8)

    if mask_np.ndim == 3:
        mask_np = mask_np.reshape(mask_np.shape[-2], mask_np.shape[-1])

    if mask_np.shape != (height, width):
        print(
            f"[smoke_eval] WARNING: mask shape {mask_np.shape} does not match target {(height, width)}; "
            "resizing for diagnostics."
        )
        if cv2 is not None:
            mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            tensor = torch.from_numpy(mask_np.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(tensor, size=(height, width), mode="nearest")
            mask_np = resized.squeeze(0).squeeze(0).to(dtype=torch.uint8).cpu().numpy()

    return mask_np


def _build_dummy_outputs(inputs: List[dict]):
    outputs = []
    for sample in inputs:
        merged_mask = sample.get("gt_mask_merged")
        image_tensor = sample.get("image")
        if image_tensor is not None and torch.is_tensor(image_tensor):
            height, width = image_tensor.shape[1:]
        else:
            height = int(sample.get("height", 1))
            width = int(sample.get("width", 1))
            height = max(height, 1)
            width = max(width, 1)

        mask_np = _as_numpy_mask(merged_mask, height, width)
        mask_tensor = torch.from_numpy(mask_np.astype(np.float32, copy=False)).unsqueeze(0)

        spatial_size = mask_tensor.shape[-2:]

        ref_seg = torch.zeros((2,) + spatial_size, dtype=torch.float32)
        ref_seg[1] = mask_tensor.squeeze(0)

        nt_label = torch.tensor([0.0, 1.0], dtype=torch.float32)

        outputs.append({"ref_seg": ref_seg, "nt_label": nt_label})
    return outputs


def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray):
    intersection = int(np.logical_and(pred_mask, gt_mask).sum())
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    return intersection, union


def _resize_to_shape(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    if cv2 is not None:
        return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    tensor = torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(height, width), mode="nearest")
    return resized.squeeze(0).squeeze(0).to(dtype=torch.uint8).cpu().numpy()


def _decode_annotation_offline(
    ann: Dict,
    height: int,
    width: int,
    *,
    original_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[np.ndarray], str]:
    seg = ann.get("segmentation")
    masks: List[np.ndarray] = []
    statuses: List[str] = []

    if isinstance(seg, dict):
        mask, status = _rle_to_mask_safe(seg, height, width, original_size=original_size)
        if mask is not None:
            masks.append(mask)
        statuses.append(status)
    elif isinstance(seg, (list, tuple)):
        if not seg:
            statuses.append("seg_missing")
        elif all(isinstance(poly, (list, tuple)) for poly in seg):
            mask, status = _poly_to_mask_safe(
                seg, height, width, original_size=original_size
            )
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


def _merge_group_offline(
    group_anns: Sequence[Dict],
    height: int,
    width: int,
    *,
    img_hw_map: Dict[int, Tuple[int, int]],
) -> Tuple[np.ndarray, str]:
    group_masks: List[np.ndarray] = []
    statuses: List[str] = []
    fallback_used = False

    for ann in group_anns:
        image_id = ann.get("image_id")
        original_hw = None
        if image_id is not None:
            try:
                original_hw = img_hw_map.get(int(image_id))
            except (TypeError, ValueError):
                original_hw = None

        mask, status = _decode_annotation_offline(
            ann,
            height,
            width,
            original_size=original_hw,
        )
        if mask is not None and mask.sum() > 0:
            group_masks.append(mask)
            statuses.append(status)
            continue

        bbox_mask, bbox_status = bbox_to_mask(
            ann.get("bbox"),
            height,
            width,
            bbox_mode=ann.get("bbox_mode", "XYWH_ABS"),
        )
        if bbox_mask is not None and bbox_mask.sum() > 0:
            group_masks.append(bbox_mask)
            statuses.append(
                "+".join([s for s in [status, bbox_status] if s]) or "bbox"
            )
            fallback_used = True
        else:
            statuses.append(status or bbox_status or "decode_fail")

    merged_mask, merged_status = merge_instance_masks(group_masks, height, width, statuses=statuses)

    if (merged_mask is None or merged_mask.sum() == 0) and group_anns:
        bbox_masks: List[np.ndarray] = []
        bbox_statuses: List[str] = []
        for ann in group_anns:
            bbox_mask, bbox_status = bbox_to_mask(
                ann.get("bbox"),
                height,
                width,
                bbox_mode=ann.get("bbox_mode"),
            )
            if bbox_mask is not None and bbox_mask.sum() > 0:
                bbox_masks.append(bbox_mask)
            bbox_statuses.append(bbox_status)
        fallback_mask, fallback_status = merge_instance_masks(
            bbox_masks,
            height,
            width,
            statuses=bbox_statuses,
        )
        if fallback_mask is not None and fallback_mask.sum() > 0:
            merged_mask = fallback_mask
            merged_status = (
                "+".join([merged_status, fallback_status])
                if merged_status
                else fallback_status
            )
            fallback_used = True

    if merged_mask is None or merged_mask.sum() == 0:
        synthetic = np.zeros((height, width), dtype=np.uint8)
        synthetic[height // 2, width // 2] = 1
        merged_mask = synthetic
        merged_status = (
            f"{merged_status}+synthetic" if merged_status else "synthetic"
        )
        fallback_used = True

    merged_mask = np.ascontiguousarray(merged_mask.astype(np.uint8, copy=False))
    merged_mask[merged_mask > 0] = 1

    status_tokens = set()
    for token in merged_status.split("+"):
        token = token.strip().lower()
        if token:
            status_tokens.add(token)
    label_map = {
        "rle": "RLE",
        "poly": "POLY",
        "bbox": "BBOX",
        "synthetic": "SYNTHETIC",
    }
    readable = " + ".join(label_map.get(tok, tok.upper()) for tok in sorted(status_tokens))
    if not readable:
        readable = "UNKNOWN"

    return merged_mask, readable + (" (fallback)" if fallback_used else "")


def _run_offline(args: argparse.Namespace) -> None:
    mapper = _ensure_mapper_preloaded()
    mapper_cache = getattr(mapper, "id_to_ann", {}) if mapper is not None else {}
    cache_source = _describe_mapper_source()
    if mapper_cache:
        print(
            f"[Offline Test] Mapper cache primed with {len(mapper_cache)} annotations (source={cache_source})."
        )
    else:
        print(
            f"[Offline Test] Mapper cache empty (source={cache_source}). Falling back to instances JSON only."
        )

    with open(args.dataset_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(args.instances_json, "r", encoding="utf-8") as f:
        instances = json.load(f)

    ann_map: Dict[int, Dict] = {
        int(ann["id"]): ann for ann in instances.get("annotations", [])
    }
    img_map: Dict[int, Dict] = {
        int(img["id"]): img for img in instances.get("images", [])
    }

    max_samples = int(args.max_samples) if args.max_samples else None
    selected = 0
    targeted_checked = 0
    synthetic_failures: List[int] = []
    missing_accumulator: Dict[int, List[int]] = {}

    for sample in dataset:
        if args.split and sample.get("split") != args.split:
            continue
        if max_samples is not None and selected >= max_samples:
            break

        image_id = int(sample.get("image_id", -1))
        ann_ids = sample.get("ann_id") or []
        if not ann_ids:
            continue

        img_meta = img_map.get(image_id, {})
        height = int(img_meta.get("height", sample.get("height", 1)))
        width = int(img_meta.get("width", sample.get("width", 1)))
        height = max(height, 1)
        width = max(width, 1)

        grouped: Dict[str, List[Dict]] = {}
        missing_ann_ids: List[int] = []
        for idx, ann_id in enumerate(ann_ids):
            ann = ann_map.get(int(ann_id))
            if not ann and mapper_cache:
                ann = mapper_cache.get(int(ann_id))
            if not ann:
                missing_ann_ids.append(int(ann_id))
                continue
            group_key = str(ann.get("id", ann_id))
            grouped.setdefault(group_key, []).append(ann)

        final_mask = np.zeros((height, width), dtype=np.uint8)
        readable_tokens: List[str] = []
        used_synthetic = False
        for group_key, group_anns in grouped.items():
            group_mask, mode_label = _merge_group_offline(
                group_anns,
                height,
                width,
                img_hw_map=img_map,
            )
            final_mask = np.maximum(final_mask, group_mask)
            readable_tokens.append(mode_label)

        if final_mask.sum() == 0:
            synthetic = np.zeros((height, width), dtype=np.uint8)
            synthetic[height // 2, width // 2] = 1
            final_mask = synthetic
            readable_tokens.append("SYNTHETIC (fallback)")
            used_synthetic = True
            if missing_ann_ids:
                print(
                    f"[Offline Test] Missing annotations for {missing_ann_ids} in sample {image_id}; using synthetic fallback."
                )

        final_mask = (final_mask > 0).astype(np.uint8)
        mask_sum = int(final_mask.sum())
        readable_summary = " | ".join(readable_tokens) if readable_tokens else "UNKNOWN"

        print(
            f"✅ sample {image_id} | ann_ids {ann_ids} | mode: {readable_summary} | mask.shape {final_mask.shape} | sum = {mask_sum}"
        )

        if not sample.get("no_target", False):
            assert final_mask.shape == (height, width), "Mask shape mismatch"
            assert mask_sum > 0, "Expected non-empty mask for targeted sample"
            targeted_checked += 1

        if missing_ann_ids:
            missing_accumulator[image_id] = missing_ann_ids
        if used_synthetic and not sample.get("no_target", False):
            synthetic_failures.append(image_id)

        selected += 1

    if synthetic_failures and not args.allow_synthetic:
        print(
            "[Offline Test] FAILURE: synthetic fallback used for samples "
            f"{synthetic_failures}. Re-run with real instances.json or use --allow-synthetic to override."
        )
        raise SystemExit(2)

    summary = (
        f"[Offline Test] All {targeted_checked} targeted samples passed (shape match & non-empty)"
    )
    if synthetic_failures:
        summary += f" | synthetic allowed for {len(synthetic_failures)} samples"
    print(summary)

    if missing_accumulator:
        print(
            "[Offline Test] Missing ann_id lookup summary: "
            + "; ".join(
                f"image {img}: {ids}" for img, ids in sorted(missing_accumulator.items())
            )
        )


def _run_unit_check(args: argparse.Namespace) -> None:
    mapper_cls = globals().get("RefCOCOMapper")
    if mapper_cls is None:
        print(
            "[Unit Check] RefCOCOMapper is unavailable (Detectron2 not installed); skipping."
        )
        return

    try:
        from detectron2.structures import BoxMode
    except ModuleNotFoundError:
        print(
            "[Unit Check] Detectron2 structures could not be imported; skipping."
        )
        return

    import tempfile

    try:
        from PIL import Image
    except ImportError:
        Image = None  # type: ignore

    height = 32
    width = 32
    polygon = [8, 8, 24, 8, 24, 24, 8, 24]

    mapper = mapper_cls(
        is_train=False,
        tfm_gens=[],
        image_format="RGB",
        bert_type="bert-base-uncased",
        max_tokens=8,
        merge=True,
        preload_only=True,
    )
    mapper.id_to_ann = {}
    mapper.tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=True: [101, 102]
    )
    mapper.preload_only = False

    annotation = {
        "id": 1,
        "image_id": 1,
        "segmentation": [polygon],
        "bbox": [8, 8, 16, 16],
        "bbox_mode": int(BoxMode.XYWH_ABS),
        "category_id": 1,
    }
    mapper.id_to_ann[1] = annotation

    dataset_dict = {
        "file_name": "",
        "image_id": 1,
        "height": height,
        "width": width,
        "annotations": [annotation.copy()],
        "ann_ids": [1],
        "dataset_name": "miami2025_unit",
        "sentence": "unit test object",
        "sentences": [{"sent": "unit test object"}],
        "no_target": False,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "unit.png")
        if Image is not None:
            img = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
            img.save(image_path)
        elif cv2 is not None:
            cv2.imwrite(image_path, np.zeros((height, width, 3), dtype=np.uint8))
        else:
            print(
                "[Unit Check] Unable to create image file (no PIL/cv2); skipping."
            )
            return
        dataset_dict["file_name"] = image_path

        result = mapper(dataset_dict)

    mask = result.get("gt_mask_merged")
    if not isinstance(mask, np.ndarray):
        raise AssertionError("Mapper did not return a numpy mask for gt_mask_merged")

    if mask.shape != (height, width):
        raise AssertionError(f"Unexpected mask shape {mask.shape}")

    if int(mask.sum()) <= 0:
        raise AssertionError("Expected non-empty mask in unit check")

    mode = result.get("mask_status", {}).get("mode", "UNKNOWN")
    print(
        f"[Unit Check] PASS | mode={mode} | mask.shape={mask.shape} | sum={int(mask.sum())}"
    )


# Trigger preload once when the module is imported to satisfy pipeline expectations.
try:
    _ensure_mapper_preloaded()
except Exception as preload_exc:  # pragma: no cover - diagnostics only
    print(f"[smoke_eval] WARNING: initial mapper preload attempt failed: {preload_exc}")


def main():
    parser = argparse.ArgumentParser(description="Miami2025 evaluator smoke test")
    parser.add_argument(
        "--config-file",
        default="configs/referring_miami2025_lqm.yaml",
        help="Path to the evaluation config file.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=5,
        help="Number of dataloader batches to run before stopping (Detectron2 mode).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run mask decoding checks without Detectron2.",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        nargs="?",
        default="datasets/miami2025.json",
        help="Miami2025 referring expressions JSON (offline mode).",
    )
    parser.add_argument(
        "--instances-json",
        type=str,
        nargs="?",
        default="datasets/instances_sample.json",
        help="COCO instances JSON with segmentations (offline mode).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to filter in offline mode (e.g., train/val).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Number of samples to inspect in offline mode.",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic fallback masks to pass offline validation.",
    )
    parser.add_argument(
        "--unit-check",
        action="store_true",
        help="Run a minimal pure-Python RefCOCOMapper validation and exit.",
    )
    args = parser.parse_args()

    if args.unit_check:
        _run_unit_check(args)
        return

    if args.offline:
        _run_offline(args)
        return

    if importlib.util.find_spec("detectron2") is None:
        print(
            "Detectron2 is not installed. Use '--offline' to run the Miami2025 mask diagnostics."
        )
        return

    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, build_detection_test_loader

    from gres_model.config import add_gres_config
    from gres_model.evaluation.refer_evaluation import ReferEvaluator

    # Ensure datasets are registered on import.
    import datasets.register_miami2025  # noqa: F401

    cfg = get_cfg()
    add_gres_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = ""
    cfg.freeze()

    dataset_name = cfg.DATASETS.TEST[0]
    dataset = DatasetCatalog.get(dataset_name)
    dataset_key = dataset_name.split("_")[0] if "_" in dataset_name else dataset_name
    dataset_split = dataset_name.split("_")[-1] if "_" in dataset_name else "val"
    print(f"[{dataset_key}] Built {len(dataset)} samples for split '{dataset_split}'")

    data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = ReferEvaluator(dataset_name=dataset_name, distributed=False)

    evaluator.reset()
    data_iter: Iterable[List[dict]] = iter(data_loader)
    try:
        first_inputs = next(data_iter)
    except StopIteration:
        print(f"Dataset '{dataset_name}' produced no samples. Nothing to evaluate.")
        return

    if not first_inputs:
        print(f"Dataset '{dataset_name}' yielded an empty batch.")
        return

    first_sample = first_inputs[0]
    image_tensor = first_sample.get("image")
    if image_tensor is not None and torch.is_tensor(image_tensor):
        height, width = image_tensor.shape[1:]
    else:
        height = int(first_sample.get("height", 1))
        width = int(first_sample.get("width", 1))
        height = max(height, 1)
        width = max(width, 1)
    first_mask_np = _as_numpy_mask(first_sample.get("gt_mask_merged"), height, width)
    print(f"✅ gt_mask_merged: {type(first_sample.get('gt_mask_merged'))}")
    print(f"✅ mask shape: {first_mask_np.shape}")
    print(f"✅ mask sum: {int(first_mask_np.sum())}")

    batches_processed = 0
    for idx, inputs in enumerate(itertools.chain([first_inputs], data_iter)):
        outputs = _build_dummy_outputs(inputs)
        evaluator.process(inputs, outputs)

        for sample, output in zip(inputs, outputs):
            image_tensor = sample.get("image")
            sample_height = sample.get("height")
            sample_width = sample.get("width")
            if image_tensor is not None and torch.is_tensor(image_tensor):
                h_default, w_default = image_tensor.shape[1:3]
            else:
                h_default = w_default = 1
            height = int(sample_height) if isinstance(sample_height, (int, float)) and int(sample_height) > 0 else h_default
            width = int(sample_width) if isinstance(sample_width, (int, float)) and int(sample_width) > 0 else w_default

            gt_mask_np = _as_numpy_mask(sample.get("gt_mask_merged"), height, width)
            pred_mask_np = output["ref_seg"].argmax(dim=0).detach().cpu().numpy().astype(np.uint8)
            if pred_mask_np.shape != gt_mask_np.shape:
                pred_mask_np = _resize_to_shape(pred_mask_np, gt_mask_np.shape[0], gt_mask_np.shape[1])

            intersection, union = _compute_iou(pred_mask_np, gt_mask_np)
            if union > 0:
                iou_score = intersection / union
                iou_msg = f"IoU={iou_score:.3f} (I={intersection}, U={union})"
            else:
                iou_msg = f"IoU undefined (I={intersection}, U={union})"
            img_identifier = sample.get("image_id", "<unknown>")
            print(f"✅ img {img_identifier}: pred shape {pred_mask_np.shape}, gt shape {gt_mask_np.shape}, {iou_msg}")

        sources = [sample.get("source", "<none>") for sample in inputs]
        print(f"Batch {idx + 1}: source distribution {dict(Counter(sources))}")
        batches_processed += 1
        if batches_processed >= args.max_iters:
            break

    results = evaluator.evaluate() or {}
    print("Smoke eval done. Result keys:", list(results.keys()))


if __name__ == "__main__":
    main()
