#!/usr/bin/env python3
"""Minimal smoke test to ensure Miami2025 loader preserves `source`."""

import os


def main():
    try:
        from detectron2.config import get_cfg
        from detectron2.data import build_detection_test_loader
    except ImportError:
        print("detectron2 is not available; skipping loader check.")
        return

    from detectron2.projects.deeplab import add_deeplab_config

    from gres_model import RefCOCOMapper, add_maskformer2_config, add_refcoco_config
    from gres_model.config import add_gres_config

    # Ensure datasets are registered
    import datasets.register_miami2025  # noqa: F401

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_refcoco_config(cfg)
    add_gres_config(cfg)

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "referring_miami2025_lqm.yaml")
    cfg.merge_from_file(os.path.realpath(config_path))
    cfg.freeze()

    dataset_name = cfg.DATASETS.TEST[0]
    mapper = RefCOCOMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    first_batch = next(iter(data_loader))
    if not first_batch:
        print("Received empty batch from loader.")
        return

    first_sample = first_batch[0]
    print("Sample keys:", sorted(first_sample.keys()))
    print("Sample source:", first_sample.get("source"))


if __name__ == "__main__":
    main()
