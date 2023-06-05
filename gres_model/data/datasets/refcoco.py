import contextlib
import io
import logging
import numpy as np
import os
import random
import copy
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

"""
This file contains functions to parse RefCOCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_refcoco_json"]


def load_refcoco_json(refer_root, dataset_name, splitby, split, image_root, extra_annotation_keys=None, extra_refer_keys=None):

    if dataset_name == 'refcocop':
        dataset_name = 'refcoco+'
    if dataset_name == 'refcoco' or dataset_name == 'refcoco+':
        splitby == 'unc'
    if dataset_name == 'refcocog':
        assert splitby == 'umd' or splitby == 'google'

    dataset_id = '_'.join([dataset_name, splitby, split])

    from .refer import REFER
    logger.info('Loading dataset {} ({}-{}) ...'.format(dataset_name, splitby, split))
    logger.info('Refcoco root: {}'.format(refer_root))
    timer = Timer()
    refer_root = PathManager.get_local_path(refer_root)
    with contextlib.redirect_stdout(io.StringIO()):
        refer_api = REFER(data_root=refer_root,
                        dataset=dataset_name,
                        splitBy=splitby)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(dataset_id, timer.seconds()))

    ref_ids = refer_api.getRefIds(split=split)
    img_ids = refer_api.getImgIds(ref_ids)
    refs = refer_api.loadRefs(ref_ids)
    imgs = [refer_api.loadImgs(ref['image_id'])[0] for ref in refs]
    anns = [refer_api.loadAnns(ref['ann_id'])[0] for ref in refs]
    imgs_refs_anns = list(zip(imgs, refs, anns))

    logger.info("Loaded {} images, {} referring objects in RefCOCO format from {}".format(len(img_ids), len(ref_ids), dataset_id))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])
    ref_keys = ["raw"] + (extra_refer_keys or [])

    num_instances_without_valid_segmentation = 0

    eval_mode = (split != 'train')

    for (img_dict, ref_dict, anno_dict) in imgs_refs_anns:
        record = {}
        record["source"] = 'refcoco'
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        # Check that information of image, ann and ref match each other
        # This fails only when the data parsing logic or the annotation file is buggy.

        assert ref_dict['image_id'] == image_id
        assert anno_dict["image_id"] == image_id
        assert ref_dict['image_id'] == image_id
        assert ref_dict['ann_id'] == anno_dict['id']
        assert ref_dict['split'] == split

        # Process segmentation mask / bounding box

        obj = {key: anno_dict[key] for key in ann_keys if key in anno_dict}
        obj["bbox_mode"] = BoxMode.XYWH_ABS
        if "bbox" in obj and len(obj["bbox"]) == 0:
            raise ValueError(
                f"One annotation of image {image_id} contains empty 'bbox' value! "
                "This json does not have valid COCO format."
            )

        segm = anno_dict.get("segmentation", None)
        assert segm  # either list[list[float]] or dict(RLE)
        if isinstance(segm, dict):
            if isinstance(segm["counts"], list):
                # convert to compressed RLE
                segm = mask_util.frPyObjects(segm, *segm["size"])
        else:
            # filter out invalid polygons (< 3 points)
            segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
            if len(segm) == 0:
                num_instances_without_valid_segmentation += 1
                continue  # ignore this instance
        obj["segmentation"] = segm

        record["annotations"] = [obj]

        # Process referring expressions
        sents = ref_dict['sentences']
        for sent in sents:
            ref_record = record.copy()
            ref = {key: sent[key] for key in ref_keys if key in sent}
            ref_record["sentence"] = ref
            dataset_dicts.append(ref_record)

    # Debug mode
    # return dataset_dicts[:100]


    return dataset_dicts

if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys

    REFCOCO_PATH = '_'
    REFCOCO_DATASET = 'refcoco'
    REFCOCO_SPLITBY = 'unc'
    REFCOCO_SPLIT = 'val'
    COCO_TRAIN_2014_IMAGE_ROOT = '_'

    logger = setup_logger(name=__name__)

    dicts = load_refcoco_json(REFCOCO_PATH, REFCOCO_DATASET, REFCOCO_SPLITBY, REFCOCO_SPLIT, COCO_TRAIN_2014_IMAGE_ROOT)
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts[:10]:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata={})
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
