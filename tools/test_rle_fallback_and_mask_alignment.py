"""Temporary smoke test for RefCOCOMapper mask synthesis."""

import importlib.machinery
import importlib.util
import numpy as np
import sys
import types

MODULE_PATH = "gres_model/data/dataset_mappers/refcoco_mapper.py"


def _install_stubs():
    if "detectron2.config" not in sys.modules:
        detectron2_root = types.ModuleType("detectron2")
        detectron2_root.__spec__ = importlib.machinery.ModuleSpec("detectron2", loader=None)

        config_mod = types.ModuleType("detectron2.config")
        config_mod.__spec__ = importlib.machinery.ModuleSpec("detectron2.config", loader=None)

        def configurable(func=None, **_kwargs):
            if func is None:
                def decorator(f):
                    return f
                return decorator
            return func

        config_mod.configurable = configurable

        data_mod = types.ModuleType("detectron2.data")
        data_mod.__spec__ = importlib.machinery.ModuleSpec("detectron2.data", loader=None)
        detection_utils_mod = types.ModuleType("detectron2.data.detection_utils")
        detection_utils_mod.__spec__ = importlib.machinery.ModuleSpec("detectron2.data.detection_utils", loader=None)
        transforms_mod = types.ModuleType("detectron2.data.transforms")
        transforms_mod.__spec__ = importlib.machinery.ModuleSpec("detectron2.data.transforms", loader=None)

        detection_utils_mod.read_image = lambda path, format=None: np.zeros((1, 1, 3), dtype=np.uint8)
        detection_utils_mod.check_image_size = lambda dataset_dict, image: None
        detection_utils_mod.transform_instance_annotations = lambda obj, transforms, image_shape: obj
        detection_utils_mod.annotations_to_instances = lambda annos, image_shape: types.SimpleNamespace(
            gt_masks=types.SimpleNamespace(polygons=[], get_bounding_boxes=lambda: None),
            image_size=image_shape,
        )

        transforms_mod.Resize = lambda size: None
        transforms_mod.apply_transform_gens = lambda gens, image: (
            image,
            types.SimpleNamespace(apply_segmentation=lambda seg: seg),
        )

        structures_mod = types.ModuleType("detectron2.structures")
        structures_mod.__spec__ = importlib.machinery.ModuleSpec("detectron2.structures", loader=None)

        class _BoxMode:
            XYXY_ABS = 0

            @staticmethod
            def convert(bbox, from_mode, to_mode):
                return bbox

        structures_mod.BoxMode = _BoxMode

        detectron2_root.config = config_mod
        detectron2_root.data = data_mod
        detectron2_root.structures = structures_mod

        sys.modules["detectron2"] = detectron2_root
        sys.modules["detectron2.config"] = config_mod
        sys.modules["detectron2.data"] = data_mod
        sys.modules["detectron2.data.detection_utils"] = detection_utils_mod
        sys.modules["detectron2.data.transforms"] = transforms_mod
        sys.modules["detectron2.structures"] = structures_mod

    if "transformers" not in sys.modules:
        transformers_mod = types.ModuleType("transformers")
        transformers_mod.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

        class _StubTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                class _Tokenizer:
                    def encode(self, text, add_special_tokens=True):
                        return [101, 102]

                return _Tokenizer()

        transformers_mod.BertTokenizer = _StubTokenizer
        sys.modules["transformers"] = transformers_mod

    if "pycocotools.mask" not in sys.modules:
        pycocotools_root = types.ModuleType("pycocotools")
        pycocotools_root.__spec__ = importlib.machinery.ModuleSpec("pycocotools", loader=None)
        mask_mod = types.ModuleType("pycocotools.mask")
        mask_mod.__spec__ = importlib.machinery.ModuleSpec("pycocotools.mask", loader=None)

        def _frPyObjects(polygons, height, width):
            return {"height": height, "width": width}

        def _decode(obj):
            raise ValueError('stub decode failure')

        def _merge(rles):
            if isinstance(rles, list) and rles:
                return rles[0]
            return {"height": 0, "width": 0}

        mask_mod.frPyObjects = _frPyObjects
        mask_mod.decode = _decode
        mask_mod.merge = _merge

        pycocotools_root.mask = mask_mod

        sys.modules["pycocotools"] = pycocotools_root
        sys.modules["pycocotools.mask"] = mask_mod


def main():
    _install_stubs()

    spec = importlib.util.spec_from_file_location("refcoco_mapper", MODULE_PATH)
    refcoco_mapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refcoco_mapper)

    fake_ann_rle = {"segmentation": {"size": [10, 10], "counts": b"invalid"}}
    fake_ann_bbox = {"bbox": [2, 2, 5, 5], "bbox_mode": 0}
    dataset_dict = {
        "height": 10,
        "width": 10,
        "annotations": [fake_ann_rle, fake_ann_bbox],
    }

    mapper = refcoco_mapper.RefCOCOMapper.__new__(refcoco_mapper.RefCOCOMapper)
    synthesized = mapper._synthesize_mask(dataset_dict)

    assert isinstance(synthesized, np.ndarray)
    assert synthesized.dtype == np.uint8
    assert synthesized.shape == (10, 10)
    assert synthesized.sum() > 0, "mask should not be empty even if RLE decode fails"

    print(f"✅ RLE fallback works, mask sum = {int(synthesized.sum())}")
    print("✅ mask shape and dtype validated")


if __name__ == "__main__":
    main()
