PLAN FOR CODEX — Adapt ReLA to my custom dataset (miami2025)
0) Repository Context
This repository is ReLA (Referring Image Segmentation on top of Detectron2).
Target Python: 3.8.
Do not change model architecture or training engine; only add a dataset registration and a config.

Repo layout (abridged):

bash
ReLA/
  ├─ train_net.py
  ├─ configs/
  ├─ datasets/
  │   ├─ DATASET.md
  │   ├─ miami2025.json                # my text/split annotations
  │   ├─ instances_sample.json         # small COCO-style sample (schema reference)
  │   └─ 作业.docx                     # assignment (Chinese)
  └─ gres_model/
1) Goal
Enable ReLA to train/evaluate on my dataset under datasets/ by adding a Detectron2-style dataset registration + a config file.

2) Input Files and Schema
We have two JSON sources:

(A) datasets/miami2025.json — per-image referring expressions with split info.
Each entry looks like:

json

{
  "file_name": "COCO_train2014_000000516432.jpg",
  "image_id": 516432,
  "ann_id": [1675403],
  "category_id": [88],
  "ref_id": 86727,
  "split": "train",
  "sentences": [{"sent": "middle red teddy bear"}]
}
Notes:

ann_id may contain multiple instance ids; treat them as the target instances for this expression.

sentences may be an array; use the first item’s "sent" as the default referring sentence unless otherwise specified.

(B) datasets/instances_sample.json — small COCO-style sample for schema inference (the real instances.json is large and stored offline).
Structure:

json

{
  "images": [
    {"id": 516432, "file_name": "COCO_train2014_000000516432.jpg", "height": 480, "width": 640}
  ],
  "annotations": [
    {
      "id": 1675403,
      "image_id": 516432,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "category_id": 88,
      "iscrowd": 0
    }
  ],
  "categories": [{"id": 88, "name": "teddy bear"}]
}
Assume the real instances.json follows the same schema as this sample.

3) Expected Output (Detectron2 sample dict)
Each dataset item returned by the loader should be a dict like:

python

{
  "file_name": "datasets/miami2025/images/<split>/COCO_train2014_000000516432.jpg",
  "image_id": 516432,
  "height": 480,                # if missing in instances, can be None or read lazily
  "width":  640,                # same as above
  "annotations": [{
     "iscrowd": 0,
     "category_id": 88,
     "segmentation": [ ... ]    # COCO polygon(s) or RLE
  }],
  "ref_id": 86727,
  "sentence": "middle red teddy bear"
}
If ann_id contains multiple ids, merge all their polygons into a single annotations[0]["segmentation"] list (concatenate polygons).

4) Splits
Register four splits with these names:

miami2025_train

miami2025_val

miami2025_testA

miami2025_testB

Use the split field from miami2025.json to filter entries.

## 5) What to Generate (deliverables)

### (1) New file: `datasets/register_miami2025.py`
Implement the following:

#### Function: `load_miami2025_json(json_file: str, image_root: str, inst_json: str, target_split: str) -> List[dict]`
- Read **miami2025.json** and **instances_sample.json** (for schema inference only).
- Build two mapping tables:
  - `id -> annotation` from the `"annotations"` list in instances JSON.
  - `image_id -> image_meta` (if available) from the `"images"` list.
- For each entry whose `"split" == target_split"`:
  - Compose `file_name = os.path.join(image_root, target_split, file_name_from_json)`.
  - Collect polygons for all `ann_id` in this entry; merge into a single list.
  - Set `category_id = category_id[0]` if it’s a list; default to 0 if missing.
  - Fill `height` and `width` from `"images"` metadata if available; otherwise leave as `None` (Detectron2 can infer automatically).
  - Store `ref_id` and the first sentence’s `"sent"` as `"sentence"`.
- Return a list of dataset dictionaries following Detectron2’s standard format.

#### Function: `register_miami2025(root: str = "datasets/miami2025")`
- Detect the environment variable **`MIAMI_INST_JSON`** at runtime:
  - If set and the file exists, use it as the real `instances.json` path.
  - Otherwise, fall back to the local `datasets/miami2025/instances_sample.json`.
  - Print a message indicating which file is being used.
- For each split in `["train", "val", "testA", "testB"]`:
  - Register a DatasetCatalog entry with name `miami2025_<split>`.
  - Use a lambda to call `load_miami2025_json(json_file, image_root, inst_json, split)`.
  - Set metadata via MetadataCatalog (e.g. `evaluator_type="refcoco"`).
- Call `register_miami2025()` upon module import, so registration runs automatically.

#### Important:
- The file **`instances_sample.json`** exists **only** to help Codex/Copilot understand the schema of the real COCO-style file.
- During **actual training or evaluation**, the system should use the full **`instances.json`**,  
  by setting an environment variable before running:
  ```bash
  export MIAMI_INST_JSON=/absolute/path/to/instances.json
If the variable is not set, the script will print:

bash

[miami2025] Using fallback instance annotation file: datasets/miami2025/instances_sample.json
This indicates it is running in "schema reference mode" only, not for real experiments.

(2) Modify train_net.py
Add a single import near the top so dataset registration executes automatically when training or evaluation begins:

python

import datasets.register_miami2025  # noqa: F401

(3) New config: configs/referring_miami2025.yaml
Start from configs/referring_swin_tiny.yaml and modify only the dataset and output paths:

yaml

DATASETS:
  TRAIN: ("miami2025_train",)
  TEST:  ("miami2025_val",)
INPUT:
  IMAGE_SIZE: 384
  FORMAT: "RGB"
MODEL:
  META_ARCHITECTURE: "GRES"
  WEIGHTS: "swin_base_patch4_window12_384_22k.pkl"
SOLVER:
  IMS_PER_BATCH: 2
  BASE_LR: 1e-4
OUTPUT_DIR: "outputs/miami2025_swin_tiny"
Do not change other hyperparameters unless required.

(4) Optional helper: dataset sanity check
Provide a minimal check script (print-only) at the end of register_miami2025.py for quick verification:

Prints split sizes.

Displays one sample’s keys.

Shows the first sentence and corresponding image path.

Example to include under:

python

if __name__ == "__main__":
    from detectron2.data import DatasetCatalog
    for split in ["train", "val"]:
        ds = DatasetCatalog.get(f"miami2025_{split}")
        print(f"Split {split} sample count:", len(ds))
        if ds:
            print("Example keys:", ds[0].keys())
            print("Example sentence:", ds[0].get("sentence", ""))

6) Constraints / Non-goals
Do not modify gres_model/** or custom CUDA ops.

Do not add extra heavy dependencies; only use stdlib + Detectron2 utilities.

Python 3.8 compatibility.

Be idempotent (re-registering should not duplicate entries or crash).

7) Logging & Robustness
Minimal prints: when each split is loaded, print Loaded {N} samples for miami2025_<split>.

When an ann_id is missing in instances, skip it with a warning and continue.

Use os.path.join for all paths; no hard-coded separators.

8) Verification (Definition of Done)
After code generation and saving files:

A) Quick Python check

bash

python - <<'PY'
from detectron2.data import DatasetCatalog
print("train size:", len(DatasetCatalog.get("miami2025_train")))
print("val size:",   len(DatasetCatalog.get("miami2025_val")))
one = DatasetCatalog.get("miami2025_train")[0]
print("sample keys:", sorted(one.keys()))
print("file_name:", one["file_name"])
print("sentence:", one["sentence"][:80])
PY
B) Dry run (no full training)

bash

python train_net.py \
  --config-file configs/referring_miami2025.yaml \
  --num-gpus 1 --eval-only \
  MODEL.WEIGHTS swin_base_patch4_window12_384_22k.pkl
Expected:

No import/registration errors.

The model initializes and attempts evaluation (even with tiny data).

9) Runtime note about large files
instances_sample.json exists only for schema inference in GitHub.
On the training machine, update the loader to point to the real instances.json path (same schema).

10) Output Format & Style
Output full Python source for datasets/register_miami2025.py with clear English comments and type hints.

Show the exact import line to add in train_net.py.

Output the full YAML for configs/referring_miami2025.yaml.

Keep code readable, concise, and compatible with Python 3.8.

Single-sentence instruction to Codex
Follow this plan. Read datasets/miami2025.json and datasets/instances_sample.json, then generate datasets/register_miami2025.py, configs/referring_miami2025.yaml, and show the import line for train_net.py.
