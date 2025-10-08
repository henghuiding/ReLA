import time
import copy
import logging

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
from detectron2.structures import BoxMode

from transformers import BertTokenizer
from pycocotools import mask as coco_mask

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
    def _decode_annotation_mask(ann, height, width):
        seg = ann.get("segmentation")
        polygons = None
        if hasattr(seg, "polygons"):
            polygons = seg.polygons
        elif isinstance(seg, list) and seg:
            polygons = seg

        if polygons:
            try:
                rles = coco_mask.frPyObjects(polygons, height, width)
                decoded = coco_mask.decode(rles)
                if decoded.ndim == 3:
                    decoded = decoded.max(axis=2)
                decoded = np.asarray(decoded, dtype=np.uint8)
                decoded = RefCOCOMapper._resize_mask(decoded, height, width)
                if decoded is None:
                    return None
                return (decoded > 0).astype(np.uint8)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to decode polygon segmentation for ann %s: %s",
                    ann.get("id", "<unknown>"),
                    exc,
                )

        if isinstance(seg, dict) and seg.get("counts"):
            try:
                decoded = coco_mask.decode(seg)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to decode RLE segmentation for ann %s: %s",
                    ann.get("id", "<unknown>"),
                    exc,
                )
                decoded = None
            if decoded is not None:
                if decoded.ndim == 3:
                    decoded = decoded.max(axis=2)
                decoded = np.asarray(decoded, dtype=np.uint8)
                decoded = RefCOCOMapper._resize_mask(decoded, height, width)
                if decoded is None:
                    return None
                return (decoded > 0).astype(np.uint8)

        return None

    @staticmethod
    def _bbox_to_mask(ann, height, width):
        if "bbox" not in ann:
            return None

        bbox_mode = ann.get("bbox_mode", BoxMode.XYXY_ABS)
        x0, y0, x1, y1 = BoxMode.convert(ann["bbox"], bbox_mode, BoxMode.XYXY_ABS)
        x0 = max(0, min(int(np.floor(x0)), width))
        y0 = max(0, min(int(np.floor(y0)), height))
        x1 = max(0, min(int(np.ceil(x1)), width))
        y1 = max(0, min(int(np.ceil(y1)), height))
        if x1 <= x0 or y1 <= y0:
            return None

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 1
        return mask

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

        anns = dataset_dict.get("annotations", []) or []

        merged_mask = np.zeros((height, width), dtype=np.uint8)
        any_positive = False
        for ann in anns:
            mask = self._decode_annotation_mask(ann, height, width)
            if mask is None:
                mask = self._bbox_to_mask(ann, height, width)
            if mask is None:
                continue
            any_positive = any_positive or mask.sum() > 0
            merged_mask = np.maximum(merged_mask, mask)

        if not any_positive and anns:
            for ann in anns:
                mask = self._bbox_to_mask(ann, height, width)
                if mask is not None:
                    merged_mask = np.maximum(merged_mask, mask)

        merged_mask = (merged_mask > 0).astype(np.uint8)
        dataset_dict["height"], dataset_dict["width"] = height, width
        return merged_mask

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
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
        for obj in dataset_dict.pop("annotations"):
            if (obj.get("iscrowd", 0) != 0) or obj.get("empty", False):
                continue
            if ("bbox" not in obj) or ("bbox_mode" not in obj):
                inferred = _infer_bbox_from_segmentation(obj)
                if inferred is not None:
                    obj["bbox"] = inferred
                    obj["bbox_mode"] = BoxMode.XYXY_ABS
                else:
                    continue
            annos.append(utils.transform_instance_annotations(obj, transforms, image_shape))
        dataset_dict["annotations"] = annos
        instances = utils.annotations_to_instances(annos, image_shape)

        empty = dataset_dict.get("empty", False)

        if len(instances) > 0:
            assert (not empty)
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Generate masks from polygon
            h, w = instances.image_size
            assert hasattr(instances, 'gt_masks')
            gt_masks = instances.gt_masks
            gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            instances.gt_masks = gt_masks
        else:
            assert empty
            gt_masks = torch.zeros((0, image_shape[0], image_shape[1]), dtype=torch.uint8)
            instances.gt_masks = gt_masks

        merged_mask = self._synthesize_mask(dataset_dict)
        if self.is_train:
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

