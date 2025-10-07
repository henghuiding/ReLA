"""Minimal smoke test for the Miami2025 evaluator pipeline."""

import argparse
import importlib.util
from collections import Counter
from typing import List

import torch


def _build_dummy_outputs(inputs: List[dict]):
    outputs = []
    for sample in inputs:
        merged_mask = sample.get("gt_mask_merged")
        if merged_mask is None:
            image_tensor = sample.get("image")
            if image_tensor is not None:
                height, width = image_tensor.shape[1:]
            else:
                height = width = 1
            merged_mask = torch.zeros((1, height, width), dtype=torch.float32)
        merged_mask = merged_mask.to(dtype=torch.float32)
        spatial_size = merged_mask.shape[-2:]

        ref_seg = torch.zeros((2,) + spatial_size, dtype=torch.float32)
        ref_seg[1] = merged_mask.squeeze(0)

        nt_label = torch.tensor([0.0, 1.0], dtype=torch.float32)

        outputs.append({"ref_seg": ref_seg, "nt_label": nt_label})
    return outputs


def main():
    if importlib.util.find_spec("detectron2") is None:
        print("Detectron2 is not installed. Skipping Miami2025 smoke evaluation.")
        return

    from detectron2.config import get_cfg
    from detectron2.data import build_detection_test_loader

    from gres_model.config import add_gres_config
    from gres_model.evaluation.refer_evaluation import ReferEvaluator

    # Ensure datasets are registered on import.
    import datasets.register_miami2025  # noqa: F401

    parser = argparse.ArgumentParser(description="Miami2025 evaluator smoke test")
    parser.add_argument(
        "--config-file",
        default="configs/referring_miami2025_lqm.yaml",
        help="Path to the evaluation config file.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=5,
        help="Number of dataloader batches to run before stopping.",
    )
    args = parser.parse_args()

    cfg = get_cfg()
    add_gres_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = ""
    cfg.freeze()

    dataset_name = cfg.DATASETS.TEST[0]
    data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = ReferEvaluator(dataset_name=dataset_name, distributed=False)

    evaluator.reset()
    for idx, inputs in enumerate(data_loader):
        outputs = _build_dummy_outputs(inputs)
        evaluator.process(inputs, outputs)
        sources = [sample.get("source", "<none>") for sample in inputs]
        print(f"Batch {idx + 1}: source distribution {dict(Counter(sources))}")
        if idx + 1 >= args.max_iters:
            break

    results = evaluator.evaluate() or {}
    print("Smoke eval done. Result keys:", list(results.keys()))


if __name__ == "__main__":
    main()

