#!/usr/bin/env python3
"""
KITTI-360 M2P处理脚本
只处理drive_0000序列（已下载原图的序列）
"""

import os
import json
import shutil
import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Sequence


try:
    from ._dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root
except ImportError:
    from _dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root

_DATASET_NAME = 'KITTI-360'


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

# 配置
BASE_DIR = _default_dataset_root()
KITTI_DIR = _default_dataset_root()
OUTPUT_DIR = _default_processed_root()
PATCH_SIZE = 224
STRIDE = 112
MAJORITY_THRESHOLD = 0.95

LABEL_MAPPING_PATH = _default_dataset_root() / "label_mapping.json"
REMAINLABELS_PATH = _default_dataset_root() / "remainlabels.txt"

def load_labels():
    """加载remainlabels.txt"""
    labels = {}
    labels_path = REMAINLABELS_PATH
    if not labels_path.exists():
        labels_path = _metadata("raw_data", "remainlabels.txt")
    with open(labels_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                label_id = int(parts[0])
                label_name = parts[1]
                labels[label_id] = label_name
    return labels

def load_frames(split):
    """加载train/val帧列表，只返回drive_0000的帧"""
    frames_file = KITTI_DIR / "data_2d_semantics/train" / f"2013_05_28_drive_{split}_frames.txt"
    frames = []
    with open(frames_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and "drive_0000" in line:
                # 格式: data_2d_raw/.../XXXXXXXX.png data_2d_semantics/.../XXXXXXXX.png
                parts = line.split()
                if len(parts) >= 2:
                    img_path = parts[0]
                    mask_path = parts[1]
                    frames.append((img_path, mask_path))
    return frames

def setup_directories(split):
    """创建输出目录，并清理local_image避免旧内容干扰"""
    global_dir = OUTPUT_DIR / split / "global_image"
    local_dir = OUTPUT_DIR / split / "local_image"
    global_dir.mkdir(parents=True, exist_ok=True)
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    return global_dir, local_dir

def get_majority_class(mask_patch, valid_labels):
    """获取patch的多数类别"""
    unique, counts = np.unique(mask_patch, return_counts=True)
    total_pixels = mask_patch.size
    
    for label_id, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        if label_id in valid_labels:
            ratio = count / total_pixels
            if ratio >= MAJORITY_THRESHOLD:
                return int(label_id), ratio
    return None, 0

def process_split(split, valid_labels, label_names, limit_images=None, patch_size=PATCH_SIZE, stride=STRIDE, upscale_to=None, both=False):
    """处理一个数据分割：读取global_list.json，生成patch并记录坐标与标签"""
    print(f"\n{'='*50}")
    print(f"Processing split {split}")
    print(f"{'='*50}")

    # 目录与清理
    global_dir, local_dir = setup_directories(split)

    # 读取global_list.json，格式: {"items": [{"img":..., "mask":...}, ...]}
    list_path = global_dir / "global_list.json"
    if not list_path.exists():
        print("global_list.json not found, skipping this split")
        return 0, 0

    with open(list_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = data.get("items", [])
    if limit_images is not None and isinstance(limit_images, int) and limit_images > 0:
        items = items[:limit_images]

    total_images = len(items)
    processed_images = 0

    def show_progress(done, total):
        total = max(total, 1)
        bar_width = 30
        ratio = min(max(done / total, 0.0), 1.0)
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(ratio * 100)
        sys.stdout.write(f"\r[{split}] Progress [{bar}] {percent:3d}% ({done}/{total})")
        sys.stdout.flush()

    patch_count = 0
    local_label_list = []

    for it in items:
        img_name = it.get("img")
        mask_name = it.get("mask")
        if not img_name or not mask_name:
            processed_images += 1
            show_progress(processed_images, total_images)
            continue

        img_path = global_dir / img_name
        mask_path = global_dir / mask_name
        if not img_path.exists() or not mask_path.exists():
            processed_images += 1
            show_progress(processed_images, total_images)
            continue

        try:
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))
        except Exception as e:
            processed_images += 1
            show_progress(processed_images, total_images)
            print(f"\n[{split}] Skipping unreadable image or mask: {img_name}/{mask_name} ({e})")
            continue
        h, w = mask.shape[:2]

        base = Path(img_name).stem  # img_000001
        img_patch_count = 0

        # 定义方案列表：若 both=True，同步执行112/56(上采样到224)与224/112；否则只执行传入方案
        if both:
            schemes = [
                {"ps": 112, "st": 56, "up": 224},
                {"ps": 224, "st": 112, "up": None},
            ]
        else:
            schemes = [{"ps": int(patch_size), "st": int(stride), "up": upscale_to}]

        for sch in schemes:
            ps = sch["ps"]
            st = sch["st"]
            up = sch["up"]

            for y in range(0, h - ps + 1, st):
                for x in range(0, w - ps + 1, st):
                    mask_patch = mask[y:y+ps, x:x+ps]
                    label_id, ratio = get_majority_class(mask_patch, valid_labels)
                    if label_id is None:
                        continue

                    # 在使用112方案时，不记录vegetation标签
                    if ps == 112:
                        label_name = str(label_names.get(label_id, "")).lower()
                        if label_name == "vegetation":
                            continue

                    img_patch = img[y:y+ps, x:x+ps]
                    patch_img = Image.fromarray(img_patch)
                    if up and up != ps:
                        patch_img = patch_img.resize((up, up), resample=Image.BILINEAR)

                    img_patch_count += 1
                    patch_count += 1
                    patch_name = f"{base}_{img_patch_count:04d}.png"
                    patch_img.save(local_dir / patch_name)

                    local_label_list.append({
                        "name": patch_name,
                        "label": label_names[label_id],
                        "top_left": [int(x), int(y)],
                        "bottom_right": [int(x + ps - 1), int(y + ps - 1)],
                        "patch_size": int(ps),
                    })

        processed_images += 1
        show_progress(processed_images, total_images)

    if total_images > 0:
        sys.stdout.write("\n")

    with open(local_dir / "local_label.json", 'w', encoding='utf-8') as f:
        json.dump({"items": local_label_list}, f, ensure_ascii=False, indent=2)

    print(f"  Done: processed {processed_images}/{total_images} images, generated {patch_count} patches")
    return processed_images, patch_count

def main(argv: Sequence[str] | None = None):
    print("="*60)
    print("KITTI-360 M2P processing (drive_0000 only)")
    print("="*60)

    parser = argparse.ArgumentParser(description="KITTI-360 mask_to_patch")
    parser.add_argument("--split", choices=["train", "valid"], help="Process only the specified split")
    parser.add_argument("--limit-image", type=int, default=None, help="Limit the number of images to process")
    # 兼容旧参数名称（如果提供则覆盖前者）
    parser.add_argument("--limit-images", type=int, dest="limit_images_legacy", help="Alias for --limit-image")
    parser.add_argument("--patch-size", type=int, choices=[112, 224], default=224, help="Patch size in pixels")
    parser.add_argument("--stride", type=int, default=None, help="Stride in pixels. Defaults to patch_size/2")
    parser.add_argument("--both", action="store_true", help="Run both schemes: 112/56 (saved as 224) and 224/112")
    args = parser.parse_args(list(argv) if argv is not None else None)

    limit_images = args.limit_images_legacy if args.limit_images_legacy is not None else args.limit_image

    # 计算补丁大小与步长
    chosen_patch = args.patch_size
    chosen_stride = args.stride if args.stride is not None else chosen_patch // 2
    upscale_to = 224 if chosen_patch == 112 else None

    # 加载标签
    labels = load_labels()
    valid_labels = set(labels.keys())
    print(f"Valid labels: {labels}")

    total_global = 0
    total_patch = 0

    splits = [args.split] if args.split else ["train", "valid"]
    for split in splits:
        g, p = process_split(
            split,
            valid_labels,
            labels,
            limit_images=limit_images,
            patch_size=chosen_patch,
            stride=chosen_stride,
            upscale_to=upscale_to,
            both=args.both,
        )
        total_global += g
        total_patch += p

    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)
    print(f"Total: {total_global} images, {total_patch} patches")

def run(extra_args: Sequence[str] | None = None) -> None:
    main(extra_args)


if __name__ == "__main__":
    main()
