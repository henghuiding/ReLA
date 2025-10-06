"""Dataset registration for the miami2025 referring segmentation dataset."""

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

from detectron2.data import DatasetCatalog, MetadataCatalog

LOGGER = logging.getLogger(__name__)


def _load_json_file(json_path: str) -> Any:
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = dictionary.get(key, default)
    if value is None:
        return default
    return value


def _ensure_iterable(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def _resolve_annotation_path(root: str, filename: str) -> Optional[str]:
    candidates = [
        os.path.join(root, filename),
        os.path.join(os.path.dirname(root), filename),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def load_miami2025_json(
    json_file: str,
    image_root: str,
    inst_json: str,
    target_split: str,
) -> List[Dict[str, Any]]:
    """Load the miami2025 dataset in Detectron2 format."""
    dataset_entries = _load_json_file(json_file)
    if not isinstance(dataset_entries, list):
        raise ValueError(
            f"Expected a list of dataset entries in {json_file}, got {type(dataset_entries).__name__}."
        )

    instance_content = _load_json_file(inst_json)
    if not isinstance(instance_content, dict):
        raise ValueError(
            f"Expected a dict of instance annotations in {inst_json}, got {type(instance_content).__name__}."
        )

    annotations_map: Dict[int, Dict[str, Any]] = {
        ann["id"]: ann for ann in instance_content.get("annotations", [])
    }
    images_map: Dict[int, Dict[str, Any]] = {
        img["id"]: img for img in instance_content.get("images", [])
    }

    dataset_dicts: List[Dict[str, Any]] = []
    for entry in dataset_entries:
        if entry.get("split") != target_split:
            continue

        file_name = entry.get("file_name")
        if not file_name:
            LOGGER.warning("Skipping entry without file_name: %s", entry)
            continue

        image_id = entry.get("image_id")
        record: Dict[str, Any] = {
            "file_name": os.path.join(image_root, target_split, file_name),
            "image_id": image_id,
        }

        image_meta = images_map.get(image_id) if image_id is not None else None
        record["height"] = _safe_get(image_meta or {}, "height")
        record["width"] = _safe_get(image_meta or {}, "width")

        sentences = entry.get("sentences") or []
        sentence_text = ""
        if sentences:
            first_sentence = sentences[0]
            if isinstance(first_sentence, dict):
                sentence_text = first_sentence.get("sent", "")
            else:
                sentence_text = str(first_sentence)
        record["sentence"] = sentence_text
        record["ref_id"] = entry.get("ref_id")

        ann_ids = _ensure_iterable(entry.get("ann_id", []))
        merged_ann: Optional[Dict[str, Any]] = None
        for ann_id in ann_ids:
            annotation = annotations_map.get(int(ann_id)) if ann_id is not None else None
            if annotation is None:
                LOGGER.warning(
                    "[miami2025] Missing annotation id %s for image %s", ann_id, image_id
                )
                continue

            if merged_ann is None:
                merged_ann = {
                    "iscrowd": annotation.get("iscrowd", 0),
                    "category_id": annotation.get("category_id", 0),
                    "segmentation": [],
                }

            segmentation = annotation.get("segmentation")
            if segmentation:
                if isinstance(segmentation, list):
                    merged_ann["segmentation"].extend(segmentation)
                else:
                    merged_ann["segmentation"].append(segmentation)

        category = entry.get("category_id")
        if isinstance(category, list):
            category = category[0] if category else 0
        if merged_ann is not None:
            merged_ann["category_id"] = category if category is not None else merged_ann.get("category_id", 0)
            record["annotations"] = [merged_ann]
        else:
            record["annotations"] = []

        dataset_dicts.append(record)

    print(f"Loaded {len(dataset_dicts)} samples for miami2025_{target_split}.")
    return dataset_dicts


def register_miami2025(root: str = "datasets/miami2025") -> None:
    """Register the miami2025 dataset splits with Detectron2."""
    dataset_json = _resolve_annotation_path(root, "miami2025.json")
    if dataset_json is None:
        raise FileNotFoundError(
            f"Could not locate miami2025.json relative to root '{root}'."
        )

    env_inst_json = os.getenv("MIAMI_INST_JSON")
    inst_json: Optional[str] = None
    if env_inst_json and os.path.isfile(env_inst_json):
        inst_json = env_inst_json
    elif env_inst_json:
        LOGGER.warning(
            "[miami2025] MIAMI_INST_JSON is set but file not found: %s", env_inst_json
        )

    if inst_json is None:
        fallback = _resolve_annotation_path(root, "instances_sample.json")
        if fallback is None:
            fallback = _resolve_annotation_path(root, "instances.json")
        if fallback is None:
            raise FileNotFoundError(
                "Could not locate instances annotation file (sample or real)."
            )
        inst_json = fallback
        print(f"[miami2025] Using fallback instance annotation file: {inst_json}")

    image_root = os.path.join(root, "images")

    available_categories: List[str] = []
    try:
        instance_content = _load_json_file(inst_json)
        available_categories = [
            cat["name"]
            for cat in instance_content.get("categories", [])
            if "name" in cat
        ]
    except (OSError, json.JSONDecodeError):
        LOGGER.warning(
            "[miami2025] Unable to parse categories from %s", inst_json
        )

    split_mapping = {
        "train": "miami2025_train",
        "val": "miami2025_val",
        "testA": "miami2025_testA",
        "testB": "miami2025_testB",
    }

    for split_key, dataset_name in split_mapping.items():
        if dataset_name in DatasetCatalog.list():
            continue
        DatasetCatalog.register(
            dataset_name,
            lambda json_file=dataset_json,
            image_root=image_root,
            inst_json=inst_json,
            target_split=split_key: load_miami2025_json(
                json_file, image_root, inst_json, target_split
            ),
        )
        metadata = MetadataCatalog.get(dataset_name)
        metadata.set(
            evaluator_type="refer",
            image_root=os.path.join(image_root, split_key),
            json_file=dataset_json,
            instances_json=inst_json,
        )
        if available_categories:
            metadata.set(thing_classes=available_categories)


_default_root = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "miami2025")
register_miami2025(_default_root)


if __name__ == "__main__":
    from detectron2.data import DatasetCatalog

    for split in ["train", "val"]:
        dataset_name = f"miami2025_{split}"
        dataset = DatasetCatalog.get(dataset_name)
        print(f"Split {split} sample count: {len(dataset)}")
        if dataset:
            first_sample = dataset[0]
            print("Example keys:", list(first_sample.keys()))
            print("Example file:", first_sample.get("file_name"))
            print("Example sentence:", first_sample.get("sentence", ""))
