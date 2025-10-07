import os

from detectron2.data import DatasetCatalog

from datasets.register_miami2025 import register_miami2025 as _register_miami2025


def _detect_root() -> str:
    """Resolve the dataset root for Miami2025 registration."""
    env_root = os.getenv("MIAMI2025_DATASETS")
    if env_root:
        return env_root
    return os.getenv("DETECTRON2_DATASETS", "/autodl-tmp/rela_data")


_root = _detect_root()
_required = {"miami2025_train", "miami2025_val"}


def _ensure_registration():
    if not _required.issubset(set(DatasetCatalog.list())):
        _register_miami2025(_root)


_ensure_registration()


__all__ = ["_ensure_registration"]
