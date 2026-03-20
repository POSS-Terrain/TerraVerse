from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterator, Sequence


CODE_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = CODE_ROOT.parent.parent
DATA_ROOT = WORKSPACE_ROOT / "data"


@contextlib.contextmanager
def pushd(path: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@contextlib.contextmanager
def patched_argv(argv: Sequence[str]) -> Iterator[None]:
    old_argv = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old_argv


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_existing_dataset_layout(dataset_root: Path) -> bool:
    for folder_name in ("local_image", "local_image_final"):
        if (dataset_root / folder_name).is_dir():
            return True

    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "processed_data":
            return True
        for folder_name in ("local_image", "local_image_final"):
            if (child / folder_name).is_dir():
                return True

    return False


def describe_dataset_layout(dataset_root: Path) -> str:
    hits: list[str] = []
    for child in sorted(dataset_root.iterdir()):
        if child.is_dir():
            nested = []
            for folder_name in ("global_image", "local_image", "local_image_final", "processed_data"):
                if (child / folder_name).exists():
                    nested.append(folder_name)
            if nested:
                hits.append(f"{child.name}/({', '.join(nested)})")
            else:
                hits.append(f"{child.name}/")
    if not hits:
        return "empty"
    return ", ".join(hits)


def run_dataset_script(dataset_name: str, relative_script: str, extra_args: Sequence[str] | None = None) -> None:
    script_path = CODE_ROOT / relative_script
    if not script_path.exists():
        dataset_root = DATA_ROOT / dataset_name
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"dataset root not found for {dataset_name}: {dataset_root}")
        if not has_existing_dataset_layout(dataset_root):
            layout_desc = describe_dataset_layout(dataset_root)
            raise FileNotFoundError(
                "\n".join([
                    f"mask_to_patch script not found for {dataset_name}: {script_path}",
                    f"current dataset layout: {dataset_root} -> {layout_desc}",
                    "required input is one of: <dataset>/local_image_final, <dataset>/local_image, <dataset>/<split>/local_image_final, <dataset>/<split>/local_image",
                    "or you need to restore / port the original mask_to_patch.py for this dataset.",
                ])
            )
        print(f"[mask_to_patch] {dataset_name}: detected existing dataset layout, skip")
        return

    module_name = f"terraverse_mask_to_patch_{dataset_name.lower().replace('-', '_')}"
    module = load_module(module_name, script_path)
    if not hasattr(module, 'main'):
        raise AttributeError(f"{script_path} has no main()")

    argv = [str(script_path)]
    if extra_args:
        argv.extend(extra_args)

    with pushd(script_path.parent), patched_argv(argv):
        module.main()
