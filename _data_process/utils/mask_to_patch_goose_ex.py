#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Sequence

import numpy as np
from PIL import Image


try:
    from ._dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root
except ImportError:
    from _dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root

_DATASET_NAME = 'GOOSE-Ex'


def _default_dataset_root() -> Path:
    return _shared_dataset_root(_DATASET_NAME)


def _default_processed_root(legacy_name: str = "processed_data") -> Path:
    return _shared_processed_root(_DATASET_NAME, legacy_name)


def _metadata(*parts: str) -> Path:
    direct = _default_dataset_root().joinpath(*parts)
    if direct.exists():
        return direct
    return _shared_metadata_path(_DATASET_NAME, *parts)


def _looks_like_processed_root(path: Path) -> bool:
    return any((path / name).exists() for name in ("train", "valid", "test", "global_image"))


def _default_mapping_path() -> Path:
    return _default_dataset_root() / "goose_label_mapping.csv"


def _resolve_mapping_path(mapping_root: Path | None = None, split_name: str | None = None) -> Path:
    shared_path = _default_mapping_path()
    if shared_path.exists():
        return shared_path
    if mapping_root is not None:
        direct_path = mapping_root / "goose_label_mapping.csv"
        if direct_path.exists():
            return direct_path
        if split_name is not None:
            legacy_names = {
                "train": ("goose_2d_train", "gooseEx_2d_train"),
                "valid": ("goose_2d_val", "gooseEx_2d_val"),
                "test": ("goose_2d_test", "gooseEx_2d_test"),
            }
            for legacy_name in legacy_names.get(split_name, ()):
                legacy_path = mapping_root / legacy_name / "goose_label_mapping.csv"
                if legacy_path.exists():
                    return legacy_path
    return shared_path

DEFAULT_PATCH_SIZE = 224
DEFAULT_TARGET_SIZE = 224
DEFAULT_STRIDE = DEFAULT_PATCH_SIZE // 2
DEFAULT_THRESHOLD = 0.95


def ensure_clean_dir(path: Path, reset: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not reset:
        return
    for p in path.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_file() or sub.is_symlink():
                    sub.unlink(missing_ok=True)
            p.rmdir()


def load_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB" if im.mode in {"RGB", "RGBA"} else "L"))


def load_label_mapping(mapping_path: Path) -> Tuple[Dict[int, str], Dict[int, int]]:
    """Parse goose_label_mapping.csv into id->name and color->id maps."""
    id_to_name: Dict[int, str] = {}
    color_to_id: Dict[int, int] = {}
    with mapping_path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_name, label_key, _, hex_code = line.split(",")
            label_id = int(label_key)
            id_to_name[label_id] = class_name
            if hex_code.startswith("#") and len(hex_code) == 7:
                r = int(hex_code[1:3], 16)
                g = int(hex_code[3:5], 16)
                b = int(hex_code[5:7], 16)
                packed = (r << 16) + (g << 8) + b
                color_to_id[packed] = label_id
    return id_to_name, color_to_id


def mask_to_label_ids(mask: np.ndarray, color_to_id: Dict[int, int]) -> np.ndarray:
    if mask.ndim == 2:
        return mask.astype(np.int32)

    h, w, c = mask.shape
    if c >= 3:
        rgb = mask[:, :, :3].astype(np.int32)
        packed = (rgb[:, :, 0] << 16) + (rgb[:, :, 1] << 8) + rgb[:, :, 2]
        flat = packed.reshape(-1)
        out_flat = np.zeros_like(flat)
        for color, label_id in color_to_id.items():
            out_flat[flat == color] = label_id
        return out_flat.reshape(h, w)

    return mask[:, :, 0].astype(np.int32)


def iter_patch_origins(h: int, w: int, patch: int, stride: Optional[int]) -> Iterable[Tuple[int, int]]:
    if h < patch or w < patch:
        return []
    step = (patch // 2) if stride is None else stride
    ys = range(0, h - patch + 1, step)
    xs = range(0, w - patch + 1, step)
    for y in ys:
        for x in xs:
            yield y, x


def process_split(
    split_name: str,
    processed_root: Path,
    mapping_root: Path,
    stride: Optional[int],
    min_majority: float,
    limit_images: Optional[int],
    reset: bool,
    target_size: int,
) -> int:
    global_dir = processed_root / split_name / "global_image"
    local_dir = processed_root / split_name / "local_image"
    ensure_clean_dir(local_dir, reset=reset)

    mapping_path = _resolve_mapping_path(mapping_root, split_name)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping not found: {mapping_path}")
    id_to_name, color_to_id = load_label_mapping(mapping_path)

    global_list_path = global_dir / "global_list.json"
    if not global_list_path.exists():
        return 0
    with global_list_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    items: List[Dict[str, str]] = payload.get("items", [])

    saved: List[Dict[str, str]] = []

    total_images = len(items) if limit_images is None else min(len(items), limit_images)
    processed_images = 0

    def show_progress(done: int, total: int) -> None:
        total = max(total, 1)
        bar_width = 30
        ratio = min(max(done / total, 0.0), 1.0)
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(ratio * 100)
        sys.stdout.write(f"\r[{split_name}] Progress [{bar}] {percent:3d}% ({done}/{total})")
        sys.stdout.flush()

    for idx, entry in enumerate(items, start=1):
        if limit_images and idx > limit_images:
            break
        img_path = global_dir / entry["img"]
        mask_path = global_dir / entry["mask"]
        if not img_path.exists() or not mask_path.exists():
            processed_images += 1
            show_progress(processed_images, total_images)
            continue

        img = load_png(img_path)
        mask = load_png(mask_path)
        label_ids = mask_to_label_ids(mask, color_to_id)

        h, w = label_ids.shape
        patch_idx = 0
        base = Path(entry["img"]).stem
        patch_size = DEFAULT_PATCH_SIZE
        for y, x in iter_patch_origins(h, w, patch_size, stride):
            patch_labels = label_ids[y : y + patch_size, x : x + patch_size]
            vals, counts = np.unique(patch_labels, return_counts=True)
            best_idx = int(np.argmax(counts))
            majority_label = int(vals[best_idx])
            majority_ratio = float(counts[best_idx]) / float(patch_size * patch_size)
            if majority_ratio < min_majority:
                continue

            patch_img = img[y : y + patch_size, x : x + patch_size]
            resized = Image.fromarray(patch_img).resize((target_size, target_size), Image.BILINEAR)
            patch_idx += 1
            patch_name = f"{base}_{patch_idx:04d}.png"
            resized.save(local_dir / patch_name)

            label_name = id_to_name.get(majority_label, "undefined")
            saved.append({
                "name": patch_name,
                "label": label_name,
                "top_left": [int(x), int(y)],
                "bottom_right": [int(x + patch_size - 1), int(y + patch_size - 1)],
            })

        processed_images += 1
        show_progress(processed_images, total_images)

    if total_images > 0:
        sys.stdout.write("\n")

    (local_dir / "local_label.json").write_text(
        json.dumps({"items": saved, "count": len(saved)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return len(saved)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract GOOSE-Ex patches with label names (224 patch, stride 112 default, upsampled to 224, >95% majority)",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=_default_processed_root(),
        help="Path to processed_data root (contains train/valid/test)",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=_default_dataset_root(),
        help="Fallback mapping root; defaults to data/GOOSE-Ex and prefers data/GOOSE-Ex/goose_label_mapping.csv",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Process a single split (train/valid/test)",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "valid", "test"],
        help="Which splits to process",
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Stride for sliding window; default=112 when patch=224")
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE, help="Output patch size after upsampling")
    parser.add_argument("--min-majority", type=float, default=DEFAULT_THRESHOLD, help="Keep patch if majority >= this")
    parser.add_argument("--limit-image", type=int, default=None, help="Limit number of source images per split (single form)")
    parser.add_argument("--limit-images", type=int, default=None, help="Limit number of source images per split (plural form)")
    parser.add_argument("--no-reset", action="store_true", help="Do not clear existing local_image contents")
    args = parser.parse_args(list(argv) if argv is not None else None)

    split_names = [args.split] if args.split else args.splits
    limit_images = args.limit_image if args.limit_image is not None else args.limit_images
    total = 0
    for split in split_names:
        count = process_split(
            split_name=split,
            processed_root=args.processed_root,
            mapping_root=args.raw_root,
            stride=args.stride,
            min_majority=args.min_majority,
            limit_images=limit_images,
            reset=(not args.no_reset),
            target_size=args.target_size,
        )
        print(f"[{split}] saved {count} patches -> {args.processed_root / split / 'local_image'}")
        total += count

    print(f"Done. Total patches: {total}")


def run(extra_args: Sequence[str] | None = None) -> None:
    main(extra_args)


if __name__ == "__main__":
    main()
