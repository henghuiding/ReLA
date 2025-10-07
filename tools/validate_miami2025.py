import os

from detectron2.data import DatasetCatalog

from datasets.register_miami2025 import register_miami2025


def check(name):
    ds = DatasetCatalog.get(name)
    miss_img = miss_keys = 0
    for r in ds:
        if not os.path.exists(r["file_name"]):
            miss_img += 1
        if "annotations" not in r or not r["annotations"]:
            miss_keys += 1
        else:
            a = r["annotations"][0]
            for k in ["segmentation", "bbox", "bbox_mode", "category_id"]:
                if k not in a:
                    miss_keys += 1
                    break
    print(f"[{name}] total={len(ds)} missing_images={miss_img} bad_annos={miss_keys}")


if __name__ == "__main__":
    required = ("miami2025_train", "miami2025_val")

    if not set(required).issubset(set(DatasetCatalog.list())):
        register_miami2025()

    for name in required:
        try:
            check(name)
        except KeyError:
            print(f"[{name}] not registered, skip")
