import json
import os
import sys
import tempfile
import types
import importlib.machinery
import unittest
from unittest import mock

import numpy as np
from pathlib import Path

try:  # pragma: no cover - optional dependency for tests
    from detectron2.structures import BoxMode  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - lightweight stub for CI
    structures = types.ModuleType("detectron2.structures")

    detectron2_module = sys.modules.setdefault("detectron2", types.ModuleType("detectron2"))
    detectron2_module.__spec__ = importlib.machinery.ModuleSpec("detectron2", loader=None)
    structures.__spec__ = importlib.machinery.ModuleSpec("detectron2.structures", loader=None)

    class _BoxModeStub:
        XYXY_ABS = "XYXY_ABS"
        XYWH_ABS = "XYWH_ABS"

        @staticmethod
        def convert(bbox, from_mode, to_mode):
            if from_mode in (0, "XYXY_ABS", _BoxModeStub.XYXY_ABS):
                return bbox
            if from_mode in (1, "XYWH_ABS", _BoxModeStub.XYWH_ABS):
                x0, y0, w, h = bbox
                return [x0, y0, x0 + w, y0 + h]
            raise ValueError("Unsupported BoxMode conversion in stub")

    structures.BoxMode = _BoxModeStub
    detectron2_module = sys.modules.setdefault("detectron2", detectron2_module) if "detectron2_module" in locals() else sys.modules["detectron2"]
    sys.modules["detectron2.structures"] = structures
    BoxMode = _BoxModeStub

if "detectron2.config" not in sys.modules:  # pragma: no cover - minimal stub
    config_module = types.ModuleType("detectron2.config")
    config_module.__spec__ = importlib.machinery.ModuleSpec("detectron2.config", loader=None)

    def configurable(func=None, **_kwargs):
        if func is None:
            def decorator(inner):
                return inner

            return decorator
        return func

    config_module.configurable = configurable
    sys.modules["detectron2.config"] = config_module

if "detectron2.data" not in sys.modules:  # pragma: no cover - minimal stub
    data_module = types.ModuleType("detectron2.data")
    data_module.__spec__ = importlib.machinery.ModuleSpec("detectron2.data", loader=None)

    class _MetadataCatalogStub:
        _store = {}

        @classmethod
        def get(cls, name):
            class _MetaStub:
                def __init__(self):
                    self.data = {}

                def set(self, **kwargs):
                    self.data.update(kwargs)
                    return self

            if name not in cls._store:
                cls._store[name] = _MetaStub()
            return cls._store[name]

    class _DatasetCatalogStub:
        registry = {}

        @classmethod
        def register(cls, name, func):
            cls.registry[name] = func

        @classmethod
        def get(cls, name):
            if name not in cls.registry:
                raise KeyError(name)
            return cls.registry[name]()

        @classmethod
        def list(cls):
            return list(cls.registry.keys())

    def _apply_transform_gens(tfm_gens, image):
        class _TransformStub:
            def apply_segmentation(self, mask):
                return mask

        return image, _TransformStub()

    data_module.MetadataCatalog = _MetadataCatalogStub
    data_module.DatasetCatalog = _DatasetCatalogStub
    data_module.detection_utils = types.SimpleNamespace(
        read_image=lambda path, format=None: np.zeros((1, 1, 3), dtype=np.uint8),
        check_image_size=lambda dataset_dict, image: None,
        annotations_to_instances=lambda annotations, image_shape: mock.Mock(
            gt_masks=mock.Mock(polygons=[], get_bounding_boxes=lambda: mock.Mock()),
            image_size=image_shape,
        ),
    )
    data_module.transforms = types.SimpleNamespace(apply_transform_gens=_apply_transform_gens)
    sys.modules["detectron2.data"] = data_module

import importlib.util

MODULE_PATH = Path(__file__).resolve().parents[1] / "gres_model" / "data" / "dataset_mappers" / "refcoco_mapper.py"
spec = importlib.util.spec_from_file_location("refcoco_mapper_module", MODULE_PATH)
refcoco_module = importlib.util.module_from_spec(spec)
if "gres_model.utils.mask_ops" not in sys.modules:
    gres_model_pkg = types.ModuleType("gres_model")
    gres_model_pkg.__path__ = []
    gres_model_pkg.__spec__ = importlib.machinery.ModuleSpec("gres_model", loader=None, is_package=True)
    sys.modules["gres_model"] = gres_model_pkg
    utils_pkg = types.ModuleType("gres_model.utils")
    utils_pkg.__path__ = []
    utils_pkg.__spec__ = importlib.machinery.ModuleSpec("gres_model.utils", loader=None, is_package=True)
    sys.modules["gres_model.utils"] = utils_pkg
    mask_ops_path = Path(__file__).resolve().parents[1] / "gres_model" / "utils" / "mask_ops.py"
    mask_spec = importlib.util.spec_from_file_location("gres_model.utils.mask_ops", mask_ops_path)
    mask_module = importlib.util.module_from_spec(mask_spec)
    mask_spec.loader.exec_module(mask_module)
    sys.modules["gres_model.utils.mask_ops"] = mask_module

if "pycocotools.mask" not in sys.modules:
    mask_module = types.ModuleType("pycocotools.mask")
    mask_module.frPyObjects = lambda polygons, h, w: np.zeros((h, w), dtype=np.uint8)
    mask_module.decode = lambda rle: np.ones((1, 1), dtype=np.uint8)
    pycoco_module = types.ModuleType("pycocotools")
    pycoco_module.mask = mask_module
    sys.modules["pycocotools"] = pycoco_module
    sys.modules["pycocotools.mask"] = mask_module

spec.loader.exec_module(refcoco_module)
RefCOCOMapper = refcoco_module.RefCOCOMapper



class RefCOCOMapperInvariantTests(unittest.TestCase):
    def setUp(self):
        self.mapper = RefCOCOMapper(
            is_train=False,
            tfm_gens=[],
            image_format="RGB",
            bert_type="bert-base-uncased",
            max_tokens=4,
            merge=True,
            preload_only=True,
        )

    def test_normalize_hw_requires_trusted_source(self):
        dataset = {}
        hw = RefCOCOMapper._normalize_hw(
            [("image_tensor", (5, 7))],
            dataset_dict=dataset,
        )
        self.assertEqual(hw, (5, 7))
        self.assertTrue(dataset["_mask_assertions"]["has_hw_from_images_json"])
        self.assertTrue(dataset["_mask_assertions"]["no_1x1_anywhere"])

    def test_normalize_hw_without_trusted_source_raises(self):
        with self.assertRaises(AssertionError):
            RefCOCOMapper._normalize_hw(
                [("dataset_fields", (6, 8))],
                dataset_dict={},
            )

    def test_decode_annotation_mask_polygon_even_length(self):
        dataset = {}
        polygon = [[0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0]]
        with mock.patch.object(
            refcoco_module,
            "_poly_to_mask_safe",
            return_value=(np.ones((4, 4), dtype=np.uint8), "poly"),
        ):
            mask, status = RefCOCOMapper._decode_annotation_mask(
                {"segmentation": polygon},
                4,
                4,
                dataset_dict=dataset,
            )
        self.assertEqual(status, "poly")
        self.assertEqual(mask.shape, (4, 4))
        self.assertEqual(int(mask.sum()), 16)
        self.assertTrue(dataset["_mask_assertions"]["polygon_even_len"])
        self.assertTrue(dataset["_mask_assertions"]["decode_inputs_have_hw"])

    def test_decode_annotation_mask_polygon_invalid_length(self):
        polygon = [[0.0, 0.0, 1.0]]
        with self.assertRaises(AssertionError):
            RefCOCOMapper._decode_annotation_mask(
                {"segmentation": polygon},
                4,
                4,
                dataset_dict={},
            )

    def test_synthesize_mask_bbox_fallback_uses_true_hw(self):
        mapper = self.mapper
        mapper.id_to_ann = {
            1: {
                "id": 1,
                "image_id": 1,
                "bbox": [1.0, 1.0, 4.0, 4.0],
                "bbox_mode": BoxMode.XYXY_ABS,
                # deliberately omit segmentation to trigger bbox fallback
            }
        }
        dataset_dict = {
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "height": 8,
            "width": 8,
            "_original_hw": (8, 8),
            "_raw_annotations": [],
            "annotations": [],
            "ann_ids": [1],
            "no_target": False,
            "image_id": 1,
        }
        mask = mapper._synthesize_mask(dataset_dict)
        self.assertEqual(mask.shape, (8, 8))
        self.assertGreater(int(mask.sum()), 0)
        assertions = dataset_dict["_mask_assertions"]
        self.assertTrue(assertions["bbox_fallback_uses_true_hw"])
        self.assertTrue(assertions["coords_contract_ok"])

    def test_synthesize_mask_offline_uses_instances_metadata(self):
        mapper = self.mapper
        mapper.id_to_ann = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            inst_path = os.path.join(tmpdir, "instances.json")
            inst_payload = {
                "images": [
                    {"id": 5, "height": 10, "width": 12},
                ],
                "annotations": [
                    {
                        "id": 7,
                        "image_id": 5,
                        "bbox": [1.0, 1.0, 5.0, 5.0],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [[0.0, 0.0, 6.0, 0.0, 6.0, 6.0, 0.0, 6.0]],
                    }
                ],
            }
            with open(inst_path, "w", encoding="utf-8") as f:
                json.dump(inst_payload, f)

            dataset_dict = {
                "image": None,
                "height": None,
                "width": None,
                "_original_hw": None,
                "_raw_annotations": [],
                "annotations": [],
                "ann_ids": [7],
                "no_target": False,
                "image_id": 5,
                "inst_json": inst_path,
            }
            polygon = [[0.0, 0.0, 6.0, 0.0, 6.0, 6.0, 0.0, 6.0]]
            mask_stub = np.zeros((10, 12), dtype=np.uint8)
            mask_stub[1:6, 1:6] = 1
            with mock.patch.object(
                refcoco_module,
                "_poly_to_mask_safe",
                return_value=(mask_stub, "poly"),
            ):
                mask = mapper._synthesize_mask(dataset_dict)

        self.assertEqual(mask.shape, (10, 12))
        self.assertEqual(dataset_dict["height"], 10)
        self.assertEqual(dataset_dict["width"], 12)
        self.assertTrue(dataset_dict["_mask_assertions"]["offline_mode_has_hw"])
        self.assertTrue(dataset_dict["_mask_assertions"]["has_hw_from_images_json"])
        self.assertGreater(dataset_dict["mask_status"]["nonzero_ratio"], 0.0)
        self.assertIn(
            "10,12",
            dataset_dict["mask_audit_csv"],
        )


if __name__ == "__main__":
    unittest.main()
