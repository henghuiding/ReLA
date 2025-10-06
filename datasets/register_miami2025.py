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
from typing import List, Dict, Any
from detectron2.data import DatasetCatalog, MetadataCatalog


def _build_id_maps(inst_data: Dict[str, Any]):
    anns = inst_data.get("annotations", [])
    imgs = inst_data.get("images", [])
    id2ann = {a["id"]: a for a in anns}
    id2img = {img["id"]: img for img in imgs}
    return id2ann, id2img


def load_miami2025_json(
    json_file: str,
    image_root: str,
    inst_json: str,
    target_split: str
) -> List[Dict[str, Any]]:
    """Load miami2025 + instances COCO annotations and build Detectron2 dicts."""
    with open(json_file, "r") as f:
        miami = json.load(f)
    with open(inst_json, "r") as f:
        inst = json.load(f)

    id2ann, id2img = _build_id_maps(inst)

    # Map logical split -> actual image subdir
    split_to_dir = {
        "train": "train2014",
        "val":   "val2014",
        "testA": "val2014",
        "testB": "val2014",
    }
    subdir = split_to_dir.get(target_split, "val2014")

    data: List[Dict[str, Any]] = []
    missing_anns = 0
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
        if sent == "":
            print(f"[miami2025] Warning: empty sentence for ref_id={ref_id}")

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

        abs_path = os.path.join(image_root, subdir, file_name)
        record = {
            "file_name": abs_path,  # absolute path for safety
            "image_id": image_id,
            "height": height,
            "width":  width,
            "annotations": [{
                "iscrowd": 0,
                "category_id": cat_ids[0] if cat_ids else 0,
                "segmentation": seg_list,
            }],
            "ref_id":  ref_id,
            "sentence": sent,
        }
        data.append(record)

    print(f"[miami2025] Built {len(data)} samples for split '{target_split}' "
          f"(missing_anns={missing_anns})")
    return data


def register_miami2025(root: str = "/autodl-tmp/rela_data"):
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

    for split in ["train", "val", "testA", "testB"]:
        name = f"miami2025_{split}"
        DatasetCatalog.register(
            name,
            lambda s=split: load_miami2025_json(json_file, image_root, inst_json, s)
        )
        MetadataCatalog.get(name).set(
            image_root=image_root,
            evaluator_type="refcoco",
            json_file=json_file,
            inst_json=inst_json,
        )


# Auto-register on import
register_miami2025()


if __name__ == "__main__":
    # Quick sanity check (no model build)
    from detectron2.data import DatasetCatalog
    for split in ["train", "val"]:
        ds = DatasetCatalog.get(f"miami2025_{split}")
        print(f"[check] Split {split} count:", len(ds))
        if ds:
            print("[check] Example file:", ds[0]["file_name"])
            print("[check] Example sentence:", ds[0].get("sentence", "")[:80])
