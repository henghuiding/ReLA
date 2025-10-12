import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

# 路径
INST_JSON = "/autodl-tmp/rela_data/annotations/instances.json"
MIAMI_JSON = "/autodl-tmp/rela_data/annotations/miami2025.json"
IMG_ROOT = "/autodl-tmp/rela_data/images"

# 加载 JSON
with open(INST_JSON, "r") as f:
    instances = json.load(f)
inst_anns = {a["id"]: a for a in instances["annotations"]}
img_info = {img["id"]: img for img in instances["images"]}

with open(MIAMI_JSON, "r") as f:
    miami = json.load(f)

# 选几个样本
sample_ids = [416622, 141785, 29592, 132137]

for sid in sample_ids:
    entry = next(x for x in miami if x["image_id"] == sid)
    image_id = entry["image_id"]
    ann_ids = entry["ann_id"] if isinstance(entry["ann_id"], list) else [entry["ann_id"]]

    img_path = f"{IMG_ROOT}/{img_info[image_id]['file_name']}"
    img = np.array(Image.open(img_path).convert("RGB"))

    H, W = img.shape[:2]
    mask_total = np.zeros((H, W), dtype=np.uint8)

    for aid in ann_ids:
        ann = inst_anns.get(aid)
        if not ann: continue
        seg = ann["segmentation"]
        if isinstance(seg, list):
            rle = coco_mask.frPyObjects(seg, H, W)
        else:
            rle = seg
        mask = coco_mask.decode(rle)
        if mask.ndim == 3: mask = np.any(mask, axis=2)
        mask_total |= mask

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(mask_total, alpha=0.5, cmap="jet")
    plt.title(f"sample {sid} | ann_ids {ann_ids}")
    plt.axis("off")
    plt.show()
