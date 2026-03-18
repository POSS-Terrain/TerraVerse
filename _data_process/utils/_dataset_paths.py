from __future__ import annotations

from pathlib import Path
from typing import Iterable


CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent.parent
DATA_ROOT = WORKSPACE_ROOT / "data"

_METADATA_SUBDIRS = (
    (),
    ("metadata",),
    ("meta",),
    ("raw_data",),
)
_CODE_METADATA_SUBDIRS = (
    ("metadata",),
    ("meta",),
)
_PROCESSED_MARKERS = ("train", "valid", "test", "global_image", "local_image", "local_image_final")


def dataset_root(dataset_name: str) -> Path:
    return DATA_ROOT / dataset_name



def _iter_metadata_roots(dataset_name: str) -> Iterable[Path]:
    ds_root = dataset_root(dataset_name)
    for parts in _METADATA_SUBDIRS:
        yield ds_root.joinpath(*parts)

    for parts in _CODE_METADATA_SUBDIRS:
        yield CODE_ROOT.joinpath(*parts, dataset_name)
        yield WORKSPACE_ROOT.joinpath(*parts, dataset_name)



def metadata_path(dataset_name: str, *parts: str) -> Path:
    candidates = [root.joinpath(*parts) for root in _iter_metadata_roots(dataset_name)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return dataset_root(dataset_name).joinpath(*parts)



def processed_root(dataset_name: str, legacy_name: str = "processed_data") -> Path:
    ds_root = dataset_root(dataset_name)
    legacy_root = ds_root / legacy_name
    if legacy_root.exists():
        return legacy_root

    for marker in _PROCESSED_MARKERS:
        if (ds_root / marker).exists():
            return ds_root

    for split_name in ("train", "valid", "test"):
        split_root = ds_root / split_name
        if split_root.exists():
            return ds_root

    return legacy_root
