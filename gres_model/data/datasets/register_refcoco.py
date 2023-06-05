import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from .refcoco import load_refcoco_json
from .grefcoco import load_grefcoco_json


def register_refcoco(root):
    image_root = os.path.join(root, "images", "train2014")
    dataset_info = [
        ('refcoco', 'unc', ['train', 'val', 'testA', 'testB']),
        ('refcoco+', 'unc', ['train', 'val', 'testA', 'testB']),
        ('refcocop', 'unc', ['train', 'val', 'testA', 'testB']),
        ('refcocog', 'umd', ['train', 'val', 'test']),
        ('refcocog', 'google', ['train', 'val'])
    ]
    for name, splitby, splits in dataset_info:
        for split in splits:
            dataset_id = '_'.join([name, splitby, split])
            DatasetCatalog.register(
                dataset_id,
                lambda root=root, name=name, splitby=splitby, split=split, image_root=image_root: 
                    load_refcoco_json(root, name, splitby, split, image_root)
            )
            MetadataCatalog.get(dataset_id).set(
                evaluator_type="refer",
                dataset_name=name,
                splitby=splitby,
                split=split,
                root=root,
                image_root=image_root,
            )

def register_grefcoco(root):
    image_root = os.path.join(root, "images", "train2014")
    dataset_info = [
        ('grefcoco', 'unc', ['train', 'val', 'testA']),
    ]
    for name, splitby, splits in dataset_info:
        for split in splits:
            dataset_id = '_'.join([name, splitby, split])
            DatasetCatalog.register(
                dataset_id,
                lambda root=root, name=name, splitby=splitby, split=split, image_root=image_root: 
                    load_grefcoco_json(root, name, splitby, split, image_root)
            )
            MetadataCatalog.get(dataset_id).set(
                evaluator_type="refer",
                dataset_name=name,
                splitby=splitby,
                split=split,
                root=root,
                image_root=image_root,
            )


def merge_dataset(root, name_list, splitby, split, image_root):
    dataset_dict = []
    for name in name_list:
        if name.startswith('grefcoco'):
            dataset_dict += load_grefcoco_json(root, name, splitby, split, image_root)
        elif name.startswith('refcoco'):
            dataset_dict += load_refcoco_json(root, name, splitby, split, image_root)
    return dataset_dict


def register_grefcoco_full(root):
    image_root = os.path.join(root, "images", "train2014")
    dataset_info = [
        (('grefcoco', 'refcoco'), 'unc', ['train', 'val', 'testA']),
    ]
    for name_list, splitby, splits in dataset_info:
        for split in splits:
            dataset_id = '_'.join([name_list[0], splitby, split]) + '_full'
            DatasetCatalog.register(
                dataset_id,
                lambda root=root, name=name_list, splitby=splitby, split=split, image_root=image_root: 
                    merge_dataset(root, name, splitby, split, image_root)
            )
            MetadataCatalog.get(dataset_id).set(
                evaluator_type="refer",
                dataset_name=name_list,
                splitby=splitby,
                split=split,
                root=root,
                image_root=image_root,
            )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_refcoco(_root)
register_grefcoco(_root)
register_grefcoco_full(_root)