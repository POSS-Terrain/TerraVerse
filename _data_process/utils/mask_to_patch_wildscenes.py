#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Sequence

import numpy as np
from PIL import Image


try:
    from ._dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root
except ImportError:
    from _dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root

_DATASET_NAME = 'WildScenes'


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

DEFAULT_PATCH_SIZES = [224]
DEFAULT_STRIDE = None  # default to half patch
DEFAULT_OUTPUT_SIZE = 224
DEFAULT_THRESHOLD = 0.95


def ensure_clean_dir(path: Path, reset: bool) -> None:
    """删除已有目录并重建（当 reset 为 True），避免旧结果干扰。"""
    if reset and path.exists():
        for p in path.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                for sub in p.rglob("*"):
                    if sub.is_file() or sub.is_symlink():
                        sub.unlink(missing_ok=True)
                p.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def load_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB" if im.mode in {"RGB", "RGBA"} else "L"))


def mask_to_label_ids(mask: np.ndarray) -> np.ndarray:
    """Convert mask to label IDs.
    For WildScenes2d we prefer indexLabel PNGs which are single-channel integer IDs.
    If given a color mask, will fallback to first channel, but recommended to use indexLabel.
    """
    if mask.ndim == 2:
        return mask.astype(np.int32)
    # Fallback: treat first channel as indices (not ideal)
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


def process(
    processed_root: Path,
    patch_sizes: List[int],
    stride_override: Optional[int],
    output_size: int,
    min_majority: float,
    limit_images: Optional[int],
    reset: bool,
    saved: List[Dict[str, object]],
    base_counters: Dict[str, int],
) -> int:
    """Extract patches running multiple patch sizes sequentially per image."""
    global_dir = processed_root / "global_image"
    local_dir = processed_root / "local_image"
    ensure_clean_dir(local_dir, reset=reset)

    global_list_path = global_dir / "global_list.json"
    if not global_list_path.exists():
        return 0

    with global_list_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    items: List[Dict[str, str]] = payload.get("items", [])

    id_to_label = {
        1: "asphalt",
        2: "dirt",
        3: "mud",
        4: "water",
        5: "gravel",
        6: "other-terrain",
        7: "tree-trunk",
        8: "tree-foliage",
        9: "bush",
        10: "fence",
        11: "structure",
        12: "pole",
        13: "vehicle",
        14: "rock",
        15: "log",
        16: "other-object",
        18: "grass",
    }
    # 17: "sky",
    total_new = 0
    total_images = len(items) if limit_images is None else min(len(items), limit_images)
    processed_images = 0

    def show_progress(done: int, total: int) -> None:
        total = max(total, 1)
        bar_width = 30
        ratio = min(max(done / total, 0.0), 1.0)
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(ratio * 100)
        print(f"\r[WildScenes] Progress [{bar}] {percent:3d}% ({done}/{total})", end="")
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
        label_ids = mask_to_label_ids(mask)

        h, w = label_ids.shape
        base = Path(entry["img"]).stem
        patch_idx = base_counters.get(base, 0)

        for patch_size in patch_sizes:
            stride = (patch_size // 2) if stride_override is None else stride_override
            for y, x in iter_patch_origins(h, w, patch_size, stride):
                patch_labels = label_ids[y : y + patch_size, x : x + patch_size]
                vals, counts = np.unique(patch_labels, return_counts=True)
                best_idx = int(np.argmax(counts))
                majority_label = int(vals[best_idx])
                majority_ratio = float(counts[best_idx]) / float(patch_size * patch_size)
                if majority_ratio < min_majority:
                    continue

                label_name = id_to_label.get(majority_label)
                if label_name is None:
                    continue

                patch_img = img[y : y + patch_size, x : x + patch_size]
                patch_idx += 1
                base_counters[base] = patch_idx
                patch_name = f"{base}_{patch_idx:04d}.png"
                if patch_img.shape[0] != output_size or patch_img.shape[1] != output_size:
                    patch_img = np.array(Image.fromarray(patch_img).resize((output_size, output_size), Image.BILINEAR))
                Image.fromarray(patch_img).save(local_dir / patch_name)

                saved.append({
                    "name": patch_name,
                    "label": label_name,
                    "top_left": [int(x), int(y)],
                    "bottom_right": [int(x + patch_size - 1), int(y + patch_size - 1)],
                })
                total_new += 1

        processed_images += 1
        show_progress(processed_images, total_images)

    if total_images > 0:
        print()

    return total_new


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract WildScenes2d patches (224x224, majority filter)")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=_default_processed_root(),
        help="Path to processed_data root (contains train/valid/test)",
    )
    parser.add_argument("--split", type=str, help="Process a single split (train/valid/test)")
    parser.add_argument("--patch-sizes", nargs="*", type=int, default=DEFAULT_PATCH_SIZES, help="Patch sizes to run; default 224")
    parser.add_argument("--stride", type=int, default=None, help="Stride for sliding window; default=patch-size/2 for each size")
    parser.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE, help="Saved patch size after upsampling")
    parser.add_argument("--min-majority", type=float, default=DEFAULT_THRESHOLD, help="Keep patch if majority >= this")
    parser.add_argument("--limit-image", type=int, default=None, help="Limit number of source images (single form)")
    parser.add_argument("--limit-images", type=int, default=None, help="Limit number of source images (plural form)")
    parser.add_argument("--no-reset", action="store_true", help="Do not clear existing local_image contents")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Resolve split path if provided
    processed_root = args.processed_root / args.split if args.split else args.processed_root

    all_saved: List[Dict[str, object]] = []
    base_counters: Dict[str, int] = {}
    limit_images = args.limit_image if args.limit_image is not None else args.limit_images
    total = process(
        processed_root=processed_root,
        patch_sizes=args.patch_sizes,
        stride_override=args.stride,
        output_size=args.output_size,
        min_majority=args.min_majority,
        limit_images=limit_images,
        reset=(not args.no_reset),
        saved=all_saved,
        base_counters=base_counters,
    )
    print(f"Saved {total} patches -> {processed_root / 'local_image'}")

    # Write combined labels summary under local_image
    local_dir = processed_root / "local_image"
    (local_dir / "local_label.json").write_text(
        json.dumps({"items": all_saved, "count": len(all_saved)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Total saved {total} patches across sizes -> {processed_root / 'local_image'}")


def run(extra_args: Sequence[str] | None = None) -> None:
    main(extra_args)


if __name__ == "__main__":
    main()
