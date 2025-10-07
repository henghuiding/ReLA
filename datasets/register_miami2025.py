"""
Dataset registration for 'miami2025' using the real AutoDL data layout.

Real data layout (AutoDL):
/autodl-tmp/rela_data/
├── annotations/
│   ├── miami2025.json
│   └── instances.json
└── images/
    ├── train2014/
    └── val2014/

Notes
-----
- By default this script uses the real files under /autodl-tmp/rela_data/annotations/.
- You can override the instances path with env var MIAMI_INST_JSON:
    export MIAMI_INST_JSON=/autodl-tmp/rela_data/annotations/instances.json
- Image subdirs are mapped as:
    train -> train2014
    val   -> val2014
    testA -> val2014   (common practice: use val images)
    testB -> val2014
"""

import json
import os
import re
from typing import List, Dict, Any, Optional

from detectron2.structures import BoxMode


def _resolve_image_path(image_root, file_name_from_json, image_id=None, images_map=None):
    """Resolve the correct image path for a miami2025 sample."""
    if not file_name_from_json:
        return os.path.join(image_root, file_name_from_json or "")

    rel_path = file_name_from_json.replace("\\", "/")
    dir_part, base_name = os.path.split(rel_path)
    name_no_ext, ext = os.path.splitext(base_name)

    normalized_base = base_name
    if ext.lower() == ".jpg" and "_000000" in name_no_ext and "_" in name_no_ext:
        prefix_before_suffix = name_no_ext.rsplit("_", 1)[0]
        if "_000000" in prefix_before_suffix:
            replaced = re.sub(r"(_\d+)(\.jpg)$", r"\2", base_name)
            if replaced != base_name:
                print(
                    f"[miami2025] Normalized image name: '{base_name}' -> '{replaced}'"
                )
                normalized_base = replaced

    if dir_part:
        normalized_rel = f"{dir_part}/{normalized_base}"
    else:
        normalized_rel = normalized_base

    candidate_subdirs = []
    if "train2014" in normalized_rel:
        candidate_subdirs.append("train2014")
    if "val2014" in normalized_rel:
        candidate_subdirs.append("val2014")

    if not candidate_subdirs and images_map and image_id is not None:
        meta = images_map.get(image_id)
        if meta:
            inst_file = meta.get("file_name", "")
            if "train2014" in inst_file:
                candidate_subdirs.append("train2014")
            if "val2014" in inst_file:
                candidate_subdirs.append("val2014")

    if not candidate_subdirs:
        candidate_subdirs = ["train2014", "val2014"]

    normalized_rel = normalized_rel.lstrip("/")
    initial_candidates = []
    if "/" in normalized_rel:
        initial_candidates.append(os.path.join(image_root, normalized_rel))

    for subdir in candidate_subdirs:
        remainder = normalized_rel
        if remainder.startswith(f"{subdir}/"):
            remainder = remainder.split("/", 1)[1]
        initial_candidates.append(os.path.join(image_root, subdir, remainder))

    seen = set()
    candidates = []
    for cand in initial_candidates:
        norm_cand = os.path.normpath(cand)
        if norm_cand not in seen:
            seen.add(norm_cand)
            candidates.append(norm_cand)

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    if candidates:
        print(
            f"[miami2025] Warning: unable to find image for '{file_name_from_json}', "
            f"tried: {candidates}"
        )
        return candidates[0]

    fallback = os.path.join(image_root, normalized_rel)
    print(
        f"[miami2025] Warning: no candidate paths for '{file_name_from_json}', "
        f"fallback to {fallback}"
    )
    return fallback
from detectron2.data import DatasetCatalog, MetadataCatalog


def _build_id_maps(inst_data: Dict[str, Any]):
    anns = inst_data.get("annotations", [])
    imgs = inst_data.get("images", [])
    id2ann = {a["id"]: a for a in anns}
    id2img = {img["id"]: img for img in imgs}
    return id2ann, id2img


def _bbox_from_polys(
    polys: List[List[float]],
    height: Optional[int] = None,
    width: Optional[int] = None,
):
    """Compute an axis-aligned bbox [x0, y0, x1, y1] from segmentation polygons."""
    if not polys:
        return None

    xs: List[float] = []
    ys: List[float] = []

    for poly in polys:
        if not poly or len(poly) < 4:
            continue
        xs.extend(float(x) for x in poly[0::2])
        ys.extend(float(y) for y in poly[1::2])

    if not xs or not ys:
        return None

    if width is not None and width > 0:
        xs = [min(max(x, 0.0), float(width - 1)) for x in xs]
    if height is not None and height > 0:
        ys = [min(max(y, 0.0), float(height - 1)) for y in ys]

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    if x1 <= x0 or y1 <= y0:
        return None

    return [x0, y0, x1, y1]


def load_miami2025_json(
    json_file: str,
    image_root: str,
    inst_json: str,
    target_split: str,
    *,
    dataset_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load miami2025 + instances COCO annotations and build Detectron2 dicts."""
    with open(json_file, "r") as f:
        miami = json.load(f)
    with open(inst_json, "r") as f:
        inst = json.load(f)

    id2ann, id2img = _build_id_maps(inst)

    data: List[Dict[str, Any]] = []
    missing_anns = 0
    stats: Dict[str, int] = {}
    for item in miami:
        if item.get("split") != target_split:
            continue

        file_name = item["file_name"]
        image_id = item["image_id"]
        ann_ids  = item.get("ann_id", [])
        cat_ids  = item.get("category_id", [])
        ref_id   = item.get("ref_id", None)
        sentences = item.get("sentences")
        sent = ""
        if isinstance(sentences, list) and sentences:
            first_sent = sentences[0]
            if isinstance(first_sent, dict):
                sent = first_sent.get("sent", "")
        elif isinstance(sentences, dict):
            sent = sentences.get("sent", "")

        # Merge polygons of all referred instance ids
        seg_list = []
        for aid in ann_ids:
            ann = id2ann.get(aid)
            if ann and "segmentation" in ann:
                seg = ann["segmentation"]
                if isinstance(seg, list):
                    seg_list += seg
                else:
                    seg_list.append(seg)
            else:
                missing_anns += 1

        # Get size if present in instances.json
        height = width = None
        img_meta = id2img.get(image_id)
        if img_meta:
            height, width = img_meta.get("height", None), img_meta.get("width", None)

        bbox = _bbox_from_polys(seg_list, height=height, width=width)
        if bbox is None:
            stats["skipped_due_to_bad_bbox"] = stats.get("skipped_due_to_bad_bbox", 0) + 1
            continue

        if not sent or not sent.strip():
            stats["skipped_empty_sentence"] = stats.get("skipped_empty_sentence", 0) + 1
            continue

        cat_id = cat_ids[0] if cat_ids else 0
        ann = {
            "iscrowd": 0,
            "category_id": int(cat_id),
            "segmentation": seg_list,
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
        }

        record = {
            "file_name": _resolve_image_path(
                image_root,
                file_name,
                image_id=item.get("image_id"),
                images_map=id2img,
            ),  # absolute path for safety
            "image_id": image_id,
            "height": height,
            "width":  width,
            "annotations": [ann],
            "ref_id":  ref_id,
            "sentence": sent,
        }
        record["source"] = "miami2025"
        if dataset_name:
            record["dataset_name"] = dataset_name
        data.append(record)

    print(
        f"[miami2025] Built {len(data)} samples for split '{target_split}' "
        f"(missing_anns={missing_anns})"
    )
    print(
        f"[miami2025] skipped {stats.get('skipped_due_to_bad_bbox', 0)} invalid bboxes, "
        f"{stats.get('skipped_empty_sentence', 0)} empty sentences."
    )
    return data


def register_miami2025(
    root: str = "/autodl-tmp/rela_data",
    *,
    splits: Optional[List[str]] = None,
):
    """
    Register miami2025 splits using the real AutoDL paths by default.
    Environment override:
      - MIAMI_INST_JSON: full path to instances.json
    """
    json_file = os.path.join(root, "annotations", "miami2025.json")

    inst_json_env = os.environ.get("MIAMI_INST_JSON")
    if inst_json_env and os.path.exists(inst_json_env):
        inst_json = inst_json_env
        print(f"[miami2025] Using instances from env: {inst_json}")
    else:
        inst_json = os.path.join(root, "annotations", "instances.json")
        print(f"[miami2025] Using instances from default: {inst_json}")

    image_root = os.path.join(root, "images")

    # Helpful debug prints (kept short)
    print(f"[miami2025] json_file={json_file}")
    print(f"[miami2025] inst_json={inst_json}")
    print(f"[miami2025] image_root={image_root}")

    if splits is None:
        splits = ["train", "val"]
    else:
        splits = list(splits)

    for split in splits:
        name = f"miami2025_{split}"
        if name in DatasetCatalog.list():
            MetadataCatalog.get(name).set(
                image_root=image_root,
                evaluator_type="refcoco",
                json_file=json_file,
                inst_json=inst_json,
            )
            continue
        DatasetCatalog.register(
            name,
            lambda s=split, ds_name=name: load_miami2025_json(
                json_file,
                image_root,
                inst_json,
                s,
                dataset_name=ds_name,
            )
        )
        MetadataCatalog.get(name).set(
            image_root=image_root,
            evaluator_type="refcoco",
            json_file=json_file,
            inst_json=inst_json,
        )


def _ensure_default_registration():
    required = ("miami2025_train", "miami2025_val")
    if not set(required).issubset(set(DatasetCatalog.list())):
        register_miami2025()


# Auto-register on import if needed
_ensure_default_registration()


if __name__ == "__main__":
    # Quick sanity check (no model build)
    from detectron2.data import DatasetCatalog
    for split in ["train", "val"]:
        ds = DatasetCatalog.get(f"miami2025_{split}")
        print(f"[check] Split {split} count:", len(ds))
        if ds:
            print("[check] Example file:", ds[0]["file_name"])
            print("[check] Example sentence:", ds[0].get("sentence", "")[:80])
