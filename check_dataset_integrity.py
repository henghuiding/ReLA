import os
import json
from tqdm import tqdm
import numpy as np
from pycocotools import mask as coco_mask

# ========== 路径设置 ==========
IMG_ROOT = "/autodl-tmp/rela_data/images"
INST_JSON = "/autodl-tmp/rela_data/annotations/instances.json"
MIAMI_JSON = "/autodl-tmp/rela_data/annotations/miami2025.json"

# ========== 读取数据 ==========
print(f"[1] Loading instances from {INST_JSON}")
with open(INST_JSON, "r") as f:
    inst_data = json.load(f)
print(f"[2] Loading miami2025 from {MIAMI_JSON}")
with open(MIAMI_JSON, "r") as f:
    miami_data = json.load(f)

# 建立索引
inst_by_id = {a["id"]: a for a in inst_data["annotations"]}
img_by_id = {i["id"]: i for i in inst_data["images"]}

# ========== 辅助函数 ==========
def find_image_path(file_name: str):
    """自动匹配 train2014 或 val2014 路径"""
    direct = os.path.join(IMG_ROOT, file_name)
    if os.path.exists(direct):
        return direct
    train_path = os.path.join(IMG_ROOT, "train2014", file_name)
    if os.path.exists(train_path):
        return train_path
    val_path = os.path.join(IMG_ROOT, "val2014", file_name)
    if os.path.exists(val_path):
        return val_path
    # 某些 JSON 自带 'train2014/' 前缀
    if file_name.startswith("train2014/") and os.path.exists(os.path.join(IMG_ROOT, file_name)):
        return os.path.join(IMG_ROOT, file_name)
    if file_name.startswith("val2014/") and os.path.exists(os.path.join(IMG_ROOT, file_name)):
        return os.path.join(IMG_ROOT, file_name)
    return None

# ========== 统计项 ==========
missing_images = []
missing_anns = []
invalid_masks = []
decoded_count = 0

# ========== 检查循环 ==========
for entry in tqdm(miami_data, desc="Checking samples"):
    img_id = entry["image_id"]
    ann_ids = entry["ann_id"] if isinstance(entry["ann_id"], list) else [entry["ann_id"]]
    img_info = img_by_id.get(img_id)
    if not img_info:
        continue

    img_path = find_image_path(img_info["file_name"])
    if not img_path:
        missing_images.append(img_info["file_name"])
        continue

    H, W = img_info.get("height", 0), img_info.get("width", 0)
    if H <= 1 or W <= 1:
        continue

    for aid in ann_ids:
        ann = inst_by_id.get(aid)
        if not ann:
            missing_anns.append((img_id, aid))
            continue
        seg = ann.get("segmentation")
        if not seg:
            invalid_masks.append((img_id, aid, "empty_seg"))
            continue
        try:
            if isinstance(seg, list):  # Polygon
                rle = coco_mask.frPyObjects(seg, H, W)
            else:  # RLE
                rle = seg
            m = coco_mask.decode(rle)
            if m.ndim == 3:
                m = np.any(m, axis=2)
            if np.sum(m) == 0:
                invalid_masks.append((img_id, aid, "empty_mask"))
            decoded_count += 1
        except Exception as e:
            invalid_masks.append((img_id, aid, type(e).__name__))

# ========== 汇总输出 ==========
total_samples = len(miami_data)
print("\n==== Dataset Integrity Report ====")
print(f"Total samples checked: {total_samples}")
print(f"Missing image files:    {len(missing_images)}")
print(f"Missing ann_id in instances.json: {len(missing_anns)}")
print(f"Invalid/undecodable masks: {len(invalid_masks)}")
print(f"Successfully decoded masks: {decoded_count}")
ratio = 100 * decoded_count / total_samples
print(f"Effective sample ratio: {ratio:.2f}%")
print("==================================\n")

# ========== 详细报告输出 ==========
report_dir = "dataset_diagnosis"
os.makedirs(report_dir, exist_ok=True)
with open(os.path.join(report_dir, "missing_images.txt"), "w") as f:
    f.writelines(f"{x}\n" for x in missing_images)
with open(os.path.join(report_dir, "missing_anns.txt"), "w") as f:
    f.writelines(f"{img_id},{aid}\n" for img_id, aid in missing_anns)
with open(os.path.join(report_dir, "invalid_masks.txt"), "w") as f:
    for img_id, aid, reason in invalid_masks:
        f.write(f"{img_id},{aid},{reason}\n")

print(f"📁 Detailed logs saved in: {os.path.abspath(report_dir)}")
if missing_images:
    print("⚠️ Some COCO image files missing. Check train2014/val2014 folder names or IMG_ROOT path.")
if missing_anns:
    print("⚠️ Some ann_id not found in instances.json → likely dataset mismatch (RefCOCO vs COCO2014).")
if invalid_masks:
    print("⚠️ Some segmentation entries cannot be decoded → check 'invalid_masks.txt'.")
else:
    print("✅ All masks decodable and aligned.")
