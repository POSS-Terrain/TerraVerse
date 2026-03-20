#!/usr/bin/env python3
"""
COCOStuff数据集 Mask to Patch (M2P) 处理
1. 将images和annotations整理到global_image目录
2. 根据mask切取patch，筛选单一类别占比大于阈值的patch
3. 只保留remainlabels.txt中指定的42种标签
4. patch统一resize到224x224
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
from PIL import Image



try:
    from ._dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root
except ImportError:
    from _dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root

_DATASET_NAME = 'COCO-Stuff'


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

def load_labels(labels_path: Path) -> Dict[int, str]:
    """加载labels.txt，返回 {mask_value: label_name}"""
    label_map = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            parts = line.split(':', 1)
            mask_val = int(parts[0].strip())
            label_name = parts[1].strip()
            label_map[mask_val] = label_name
    return label_map


def load_remain_labels(remain_path: Path) -> set:
    """加载remainlabels.txt，返回需要保留的标签名称集合"""
    remain_set = set()
    with open(remain_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                remain_set.add(line.lower())
    return remain_set


def build_valid_mask_values(labels_path: Path, remain_path: Path) -> Tuple[Dict[int, str], Dict[str, str]]:
    """
    构建有效的mask值到标签名的映射
    返回: (valid_mask_to_name, name_to_id)
    """
    all_labels = load_labels(labels_path)
    remain_labels = load_remain_labels(remain_path)
	
    # 找出需要保留的mask值
    # 纠正偏移：mask值对应的label应为labels.txt中下一个label
    valid_mask_to_name = {}
    for mask_val in all_labels.keys():
        shifted_name = all_labels.get(mask_val + 1)
        if shifted_name is None:
            continue
        if shifted_name.lower() in remain_labels:
            valid_mask_to_name[mask_val] = shifted_name
	
    # 为保留的标签分配ID（从1开始，按名称排序保证一致性）
    sorted_names = sorted(set(valid_mask_to_name.values()))
    name_to_id = {name: f"{i+1:03d}" for i, name in enumerate(sorted_names)}
	
    print(f"Total labels: {len(all_labels)}, labels to keep: {len(remain_labels)}, matched labels: {len(valid_mask_to_name)}")
	
    return valid_mask_to_name, name_to_id


def setup_global_image(split_root: Path) -> List[dict]:
    """
    将images和annotations整理到global_image目录
    返回global_list的items
    """
    images_dir = split_root / "images"
    annotations_dir = split_root / "annotations"
    global_dir = split_root / "global_image"
    
    global_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    
    items = []
    for img_file in image_files:
        # img_000001.png -> 000001
        idx = img_file.replace('img_', '').replace('.png', '')
        
        # 对应的annotation文件
        ano_file = f"ano_{idx}.png"
        
        src_img = images_dir / img_file
        src_ano = annotations_dir / ano_file
        
        if not src_img.exists() or not src_ano.exists():
            continue
        
        # 目标文件名（与新版global_list保持一致）
        dst_img_name = f"img_{idx}.png"
        dst_mask_name = f"masked_{idx}.png"
        
        # 移动文件
        shutil.move(str(src_img), str(global_dir / dst_img_name))
        shutil.move(str(src_ano), str(global_dir / dst_mask_name))
        
        items.append({
            "img": dst_img_name,
            "mask": dst_mask_name
        })
    
    # 删除空的原目录
    if images_dir.exists() and not os.listdir(images_dir):
        images_dir.rmdir()
    if annotations_dir.exists() and not os.listdir(annotations_dir):
        annotations_dir.rmdir()
    
    # 保存global_list.json
    global_list = {"items": items}
    with open(global_dir / "global_list.json", 'w', encoding='utf-8') as f:
        json.dump(global_list, f, ensure_ascii=False, indent=2)
    
    return items


def iter_patches(h: int, w: int, patch: int = 224, stride: Optional[int] = None):
    """生成patch的位置"""
    stride = (patch // 2) if stride is None else stride
    for top in range(0, h - patch + 1, stride):
        for left in range(0, w - patch + 1, stride):
            yield top, left


def majority_class(mask_patch: np.ndarray, valid_mask_values: set) -> Tuple[Optional[int], float]:
    """
    计算patch中占比最大的有效类别
    返回: (mask_value, ratio) 或 (None, 0) 如果没有有效类别
    """
    # 统计每个mask值的像素数
    unique, counts = np.unique(mask_patch.flatten(), return_counts=True)
    total_pixels = mask_patch.size
    
    # 只考虑有效的mask值
    best_val = None
    best_ratio = 0.0
    
    for val, cnt in zip(unique, counts):
        if int(val) in valid_mask_values:
            ratio = cnt / total_pixels
            if ratio > best_ratio:
                best_ratio = ratio
                best_val = int(val)
    
    return best_val, best_ratio


def process_split(
    split_root: Path,
    valid_mask_to_name: Dict[int, str],
    name_to_id: Dict[str, str],
    patch_size: int = 224,
    stride: Optional[int] = None,
    min_majority: float = 0.95,
    target_size: int = 224,
    limit_images: Optional[int] = None,
) -> Tuple[int, int]:
    """处理单个split"""
    
    global_dir = split_root / "global_image"
    local_dir = split_root / "local_image"

    # 重新生成local_image目录
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取global_list
    list_path = global_dir / "global_list.json"
    with open(list_path, 'r', encoding='utf-8') as f:
        items = json.load(f).get("items", [])
    
    valid_mask_values = set(valid_mask_to_name.keys())
    label_items = []
    total_patches = 0

    total_images = len(items) if limit_images is None else min(len(items), limit_images)
    processed_images = 0

    def show_progress(done: int, total: int) -> None:
        total = max(total, 1)
        bar_width = 30
        ratio = min(max(done / total, 0.0), 1.0)
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(ratio * 100)
        sys.stdout.write(f"\rProgress [{bar}] {percent:3d}% ({done}/{total})")
        sys.stdout.flush()
    
    for i, it in enumerate(items, start=1):
        if limit_images is not None and i > limit_images:
            break
        img_name = it["img"]
        mask_name = it["mask"]
        img_path = global_dir / img_name
        mask_path = global_dir / mask_name

        if not img_path.exists() or not mask_path.exists():
            processed_images += 1
            show_progress(processed_images, total_images)
            continue
        
        # 加载图像和mask
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        
        h, w = img.shape[:2]
        
        # 获取图像序号
        img_idx = img_name.replace('img_', '').replace('.png', '')
        
        patch_count = 0
        for top, left in iter_patches(h, w, patch=patch_size, stride=stride):
            # 切取patch
            img_patch = img[top:top+patch_size, left:left+patch_size]
            mask_patch = mask[top:top+patch_size, left:left+patch_size]
            
            # 检查主要类别
            maj_val, ratio = majority_class(mask_patch, valid_mask_values)
            
            if maj_val is None or ratio < min_majority:
                continue
            
            patch_count += 1
            total_patches += 1
            
            # 生成patch文件名
            patch_name = f"img_{img_idx}_{patch_count:04d}.png"
            
            # Resize到目标尺寸并保存
            img_patch_pil = Image.fromarray(img_patch)
            if img_patch_pil.size != (target_size, target_size):
                img_patch_pil = img_patch_pil.resize((target_size, target_size), Image.LANCZOS)
            img_patch_pil.save(local_dir / patch_name)
            
            # 记录标签（包含patch坐标）
            label_name = valid_mask_to_name[maj_val]
            label_items.append({
                "name": patch_name,
                "label": label_name,
                "top_left": [int(left), int(top)],
                "bottom_right": [int(left + patch_size - 1), int(top + patch_size - 1)],
            })
        
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(items)} images, accumulated patches: {total_patches}")
    
        processed_images += 1
        show_progress(processed_images, total_images)

    # 进度条换行
    if total_images > 0:
        sys.stdout.write("\n")

    # 保存local_label.json
    local_label = {"items": label_items}
    with open(local_dir / "local_label.json", 'w', encoding='utf-8') as f:
        json.dump(local_label, f, ensure_ascii=False, indent=2)
    
    # 统计各标签数量
    label_counts = {}
    for item in label_items:
        label = item["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    return total_patches, len(label_counts)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="COCOStuff mask-to-patch processing")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_default_dataset_root(),
        help="Dataset root directory (contains labels.txt, remainlabels.txt, and processed_data)",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Process only one split (train/valid/test)",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "valid"],
        help="Splits to process",
    )
    parser.add_argument("--patch-size", type=int, default=224, help="Patch size to extract")
    parser.add_argument("--stride", type=int, default=None, help="Sliding-window stride. Defaults to half the patch size")
    parser.add_argument("--min-majority", type=float, default=0.95, help="Minimum majority-class ratio")
    parser.add_argument("--limit-image", type=int, default=None, help="Limit the number of source images to process (singular form)")
    parser.add_argument("--target-size", type=int, default=224, help="Target output patch size")
    parser.add_argument("--skip-setup", action="store_true", help="Skip global_image directory setup")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # 兼容单数/复数接口
    splits = [args.split] if args.split else args.splits
    limit_images = args.limit_image
    
    processed_root = args.data_root if _looks_like_processed_root(args.data_root) else args.data_root / "processed_data"
    labels_path = args.data_root / "labels.txt"
    if not labels_path.exists():
        labels_path = args.data_root / "raw_data" / "labels.txt"
    if not labels_path.exists():
        labels_path = Path(__file__).resolve().with_name("labels.txt")
    if not labels_path.exists():
        labels_path = _metadata("raw_data", "labels.txt")
    remain_path = args.data_root / "remainlabels.txt"
    if not remain_path.exists():
        remain_path = args.data_root / "raw_data" / "remainlabels.txt"
    if not remain_path.exists():
        remain_path = Path(__file__).resolve().with_name("remainlabels.txt")
    if not remain_path.exists():
        remain_path = _metadata("raw_data", "remainlabels.txt")
    
    print("=" * 60)
    print("COCOStuff M2P processing")
    print("=" * 60)
    
    # 构建有效标签映射
    valid_mask_to_name, name_to_id = build_valid_mask_values(labels_path, remain_path)
    
    print(f"\nValid label mapping:")
    for name, lid in sorted(name_to_id.items(), key=lambda x: x[1]):
        print(f"  {lid}: {name}")
    
    for split in splits:
        split_root = processed_root / split
        print(f"\n{'=' * 60}")
        print(f"Processing split {split}...")
        print("=" * 60)
        
        # 第一步：设置global_image目录
        if not args.skip_setup:
            if (split_root / "images").exists():
                print(f"Setting up the global_image directory...")
                items = setup_global_image(split_root)
                print(f"  Moved {len(items)} image/annotation pairs into global_image")
            else:
                print(f"  images directory not found, skipping setup")
        
        # 第二步：M2P处理
        if not (split_root / "global_image" / "global_list.json").exists():
            print(f"  global_list.json not found, skipping M2P processing")
            continue
        
        print(f"Starting M2P processing...")
        n_patches, n_labels = process_split(
            split_root=split_root,
            valid_mask_to_name=valid_mask_to_name,
            name_to_id=name_to_id,
            patch_size=args.patch_size,
            stride=args.stride,
            min_majority=args.min_majority,
            target_size=args.target_size,
            limit_images=limit_images,
        )
        print(f"  Generated patches: {n_patches}, labels covered: {n_labels}")
    
    print(f"\n{'=' * 60}")
    print("All processing completed!")
    print("=" * 60)


def run(extra_args: Sequence[str] | None = None) -> None:
    main(extra_args)


if __name__ == "__main__":
    main()
