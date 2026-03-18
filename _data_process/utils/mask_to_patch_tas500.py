#!/usr/bin/env python3
"""
TAS500数据集 Mask to Patch (M2P) 处理
仅执行 M2P：读取已有的 processed_data/<split>/global_image 下的图像与mask，切patch，筛选阈值，并保存 local_image 和标签。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
from PIL import Image



try:
    from ._dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root
except ImportError:
    from _dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root

_DATASET_NAME = 'TAS500'


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

def ensure_clean_local(local_dir: Path, reset: bool) -> None:
    """清理local_image目录避免旧结果混淆"""
    if reset and local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)


def iter_patches(h: int, w: int, patch: int = 224, stride: Optional[int] = None):
    """生成patch的位置"""
    stride = patch if stride is None else stride
    for top in range(0, h - patch + 1, stride):
        for left in range(0, w - patch + 1, stride):
            yield top, left


def load_label_map(data_root: Path) -> Dict[int, str]:
    """从 raw_data/label_mapping.json 构建 pixel_value -> label 映射。"""
    mapping_path = data_root / "raw_data" / "label_mapping.json"
    if not mapping_path.exists():
        mapping_path = _metadata("raw_data", "label_mapping.json")
    if not mapping_path.exists():
        return {}

    with open(mapping_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    label_map: Dict[int, str] = {}
    for name, info in data.items():
        pixel_val = info.get("pixel_value", -1)
        if pixel_val is None or pixel_val < 0:
            continue
        label = info.get("dataset_label") or name
        label_map[int(pixel_val)] = str(label).lower()
    return label_map


def majority_class(mask_patch: np.ndarray, label_map: Dict[int, str], background_values: Optional[set] = None) -> Tuple[Optional[str], float]:
    """返回patch中占比最大的非背景类别及其占比。"""
    background_values = {0} if background_values is None else background_values

    unique, counts = np.unique(mask_patch.flatten(), return_counts=True)
    total_pixels = mask_patch.size

    best_title = None
    best_ratio = 0.0

    for val, cnt in zip(unique, counts):
        val = int(val)
        if val in background_values:
            continue

        title = label_map.get(val)
        if title is None:
            title = f"unknown_{val}"

        ratio = cnt / total_pixels
        if ratio > best_ratio:
            best_ratio = ratio
            best_title = title

    return best_title, best_ratio


def process_split(
    split_root: Path,
    label_map: Dict[int, str],
    patch_size: int = 224,
    stride: Optional[int] = None,
    min_majority: float = 0.95,
    target_size: int = 224,
    limit_images: Optional[int] = None,
    reset: bool = True,
) -> Tuple[int, int]:
    """仅执行 M2P：读取已存在的 global_image，切 patch 并保存 local_image。"""

    global_dir = split_root / "global_image"
    local_dir = split_root / "local_image"
    ensure_clean_local(local_dir, reset=reset)

    list_path = global_dir / "global_list.json"
    if not list_path.exists():
        print(f"  global_list.json not found, skipping {split_root.name}")
        return 0, 0

    items = json.load(list_path.open('r', encoding='utf-8')).get("items", [])

    total_images = len(items) if limit_images is None else min(len(items), limit_images)
    processed_images = 0

    def show_progress(done: int, total: int) -> None:
        total = max(total, 1)
        bar_width = 30
        ratio = min(max(done / total, 0.0), 1.0)
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(ratio * 100)
        sys.stdout.write(f"\r[{split_root.name}] Progress [{bar}] {percent:3d}% ({done}/{total})")
        sys.stdout.flush()

    label_items = []
    label_set = set()
    total_patches = 0

    for i, it in enumerate(items, start=1):
        if limit_images and i > limit_images:
            break

        img_name = it["img"]
        mask_name = it["mask"]

        img_path = global_dir / img_name
        mask_path = global_dir / mask_name

        if not img_path.exists() or not mask_path.exists():
            processed_images += 1
            show_progress(processed_images, total_images)
            continue

        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        h, w = img.shape[:2]
        img_idx = img_name.replace('img_', '').replace('.png', '')

        patch_count = 0
        stride_val = patch_size if stride is None else stride
        for top, left in iter_patches(h, w, patch=patch_size, stride=stride_val):
            img_patch = img[top:top+patch_size, left:left+patch_size]
            mask_patch = mask[top:top+patch_size, left:left+patch_size]

            maj_title, ratio = majority_class(mask_patch, label_map)
            if maj_title is None or ratio < min_majority:
                continue

            patch_count += 1
            total_patches += 1

            patch_name = f"img_{img_idx}_{patch_count:04d}.png"
            img_patch_pil = Image.fromarray(img_patch)
            if img_patch_pil.size != (target_size, target_size):
                img_patch_pil = img_patch_pil.resize((target_size, target_size), Image.LANCZOS)
            img_patch_pil.save(local_dir / patch_name)

            label_items.append({
                "name": patch_name,
                "label": maj_title,
                "top_left": [int(left), int(top)],
                "bottom_right": [int(left + patch_size - 1), int(top + patch_size - 1)],
            })
            label_set.add(maj_title)

        processed_images += 1
        show_progress(processed_images, total_images)

    if total_images > 0:
        sys.stdout.write("\n")

    local_label = {"items": label_items}
    with open(local_dir / "local_label.json", 'w', encoding='utf-8') as f:
        json.dump(local_label, f, ensure_ascii=False, indent=2)

    return total_patches, len(label_set)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="TAS500 mask-to-patch processing")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_default_dataset_root(),
        help="TAS500 dataset root directory",
    )
    parser.add_argument("--split", type=str, help="Process only the specified split (train/valid/test)")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "valid", "test"],
        help="Splits to process",
    )
    parser.add_argument("--patch-size", type=int, default=224, help="Patch size to extract")
    parser.add_argument("--stride", type=int, default=None, help="Sliding-window stride. Defaults to patch size")
    parser.add_argument("--min-majority", type=float, default=0.95, help="Minimum majority-class ratio")
    parser.add_argument("--target-size", type=int, default=224, help="Target output patch size")
    parser.add_argument("--limit-image", type=int, default=None, help="Limit the number of images to process (singular form)")
    parser.add_argument("--limit-images", type=int, default=None, help="Limit the number of images to process (plural form)")
    parser.add_argument("--no-reset", action="store_true", help="Do not clear the local_image directory")
    args = parser.parse_args(list(argv) if argv is not None else None)
    
    print("=" * 60)
    print("TAS500 M2P processing")
    print("=" * 60)

    label_map = load_label_map(args.data_root)
    
    split_names = [args.split] if args.split else args.splits

    for split in split_names:
        base_root = args.data_root if _looks_like_processed_root(args.data_root) else args.data_root / "processed_data"
        split_root = base_root / split
        print(f"\n{'=' * 60}")
        print(f"Processing split {split} (M2P only, using the existing global_image)...")
        print("=" * 60)

        if not (split_root / "global_image" / "global_list.json").exists():
            print(f"  global_list.json not found, skipping M2P processing")
            continue

        n_patches, n_labels = process_split(
            split_root=split_root,
            label_map=label_map,
            patch_size=args.patch_size,
            stride=args.stride,
            min_majority=args.min_majority,
            target_size=args.target_size,
            limit_images=args.limit_image if args.limit_image is not None else args.limit_images,
            reset=(not args.no_reset),
        )
        print(f"  Generated patches: {n_patches}, labels covered: {n_labels}")
    
    print(f"\n{'=' * 60}")
    print("All processing completed!")
    print("=" * 60)


def run(extra_args: Sequence[str] | None = None) -> None:
    main(extra_args)


if __name__ == "__main__":
    main()
