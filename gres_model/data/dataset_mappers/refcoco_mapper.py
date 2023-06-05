import time
import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from transformers import BertTokenizer
from pycocotools import mask as coco_mask

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

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
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

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if (obj.get("iscrowd", 0) == 0) and (obj.get("empty", False) == False)
        ]
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

        if self.is_train:
            dataset_dict["instances"] = instances
        else:
            dataset_dict["gt_mask"] = gt_masks

        dataset_dict["empty"] = empty
        dataset_dict["gt_mask_merged"] = self._merge_masks(gt_masks) if self.merge else None


        # Language data
        sentence_raw = dataset_dict['sentence']['raw']
        attention_mask = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens

        input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

        input_ids = input_ids[:self.max_tokens]
        padded_input_ids[:len(input_ids)] = input_ids

        attention_mask[:len(input_ids)] = [1] * len(input_ids)

        dataset_dict['lang_tokens'] = torch.tensor(padded_input_ids).unsqueeze(0)
        dataset_dict['lang_mask'] = torch.tensor(attention_mask).unsqueeze(0)

        return dataset_dict
