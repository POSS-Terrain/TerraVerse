#!/usr/bin/env python3
"""
RELLIS-3D Mask to Patch Processing Script

This script processes the RELLIS-3D dataset by:
1. Reading images and label_id masks using the split files (train.lst, val.lst, test.lst)
2. Copying matched images and masks to global_image directory
3. Extracting 224x224 patches where 95%+ pixels belong to a single terrain class
4. Saving patches to local_image directory with JSON metadata

RELLIS-3D Ontology (20 classes):
0: void, 1: dirt, 3: grass, 4: tree, 5: pole, 6: water, 7: sky, 8: vehicle,
9: object, 10: asphalt, 12: building, 15: log, 17: person, 18: fence, 19: bush,
23: concrete, 27: barrier, 31: puddle, 33: mud, 34: rubble
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
from typing import Sequence


try:
    from ._dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root
except ImportError:
    from _dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root

_DATASET_NAME = 'RELLIS'


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

# Configuration
BASE_DIR = str(_default_dataset_root())
PROCESSED_DIR = str(_default_processed_root())

LABEL_MAPPING_FILE = str(_default_dataset_root() / "label_mapping.json")
REMAINLABELS_FILE = str(_default_dataset_root() / "remainlabels.txt")
if not Path(REMAINLABELS_FILE).exists():
    REMAINLABELS_FILE = str(_metadata("raw_data", "remainlabels.txt"))

PATCH_SIZE = 224
STRIDE = 112
MAJORITY_THRESHOLD = 0.95


def load_labels(filepath):
    """Load label IDs and names from remainlabels.txt"""
    labels = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                label_id = int(parts[0])
                label_name = parts[1]
                labels[label_id] = label_name
    return labels


def setup_directories(split_name, reset=False):
    """Prepare split directories under processed_data, clean local_image when reset."""
    global_dir = os.path.join(PROCESSED_DIR, split_name, 'global_image')
    local_dir = os.path.join(PROCESSED_DIR, split_name, 'local_image')
    os.makedirs(global_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)
    if reset and os.path.isdir(local_dir):
        for root, _, files in os.walk(local_dir):
            for f in files:
                try:
                    os.unlink(os.path.join(root, f))
                except Exception:
                    pass
    return {'global_dir': global_dir, 'local_dir': local_dir}


def get_majority_class(mask_patch, valid_labels):
    """
    Check if a patch has a majority class (>= 95% of pixels)
    Returns (label_id, ratio) if found, else (None, 0)
    """
    unique, counts = np.unique(mask_patch, return_counts=True)
    total_pixels = mask_patch.size
    
    for label_id, count in zip(unique, counts):
        if label_id in valid_labels:
            ratio = count / total_pixels
            if ratio >= MAJORITY_THRESHOLD:
                return label_id, ratio
    
    return None, 0


def process_split(split_name, valid_labels, label_names, dirs, limit_images=None):
    """Process a single split by reading processed_data global_list.json."""
    
    patch_count = 0
    global_count = 0
    label_stats = defaultdict(int)
    processed_images = 0

    def show_progress(done, total):
        total = max(total, 1)
        bar_width = 30
        ratio = min(max(done / total, 0.0), 1.0)
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(ratio * 100)
        sys.stdout.write(f"\r[{split_name}] Progress [{bar}] {percent:3d}% ({done}/{total})")
        sys.stdout.flush()

    # Load items from processed_data/<split>/global_image/global_list.json
    list_path = os.path.join(dirs['global_dir'], 'global_list.json')
    if not os.path.exists(list_path):
        print(f"Warning: global_list.json not found: {list_path}")
        return 0, 0, label_stats
    with open(list_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    items = payload.get('items', [])

    total_images = len(items) if not limit_images else min(len(items), int(limit_images))
    print(f"\nProcessing {split_name} split ({total_images} images from global_list.json)...")

    for idx, entry in enumerate(items, start=1):
        if limit_images and idx > limit_images:
            break
        img_path = os.path.join(dirs['global_dir'], entry.get('img', ''))
        label_path = os.path.join(dirs['global_dir'], entry.get('mask', ''))
        
        if not os.path.exists(img_path):
            if idx < 5:
                print(f"  Warning: Image not found: {img_path}")
            processed_images += 1
            show_progress(processed_images, total_images)
            continue
        if not os.path.exists(label_path):
            if idx < 5:
                print(f"  Warning: Label not found: {label_path}")
            processed_images += 1
            show_progress(processed_images, total_images)
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(label_path)
            mask_array = np.array(mask)
        except Exception as e:
            print(f"  Error loading {img_path}: {e}")
            processed_images += 1
            show_progress(processed_images, total_images)
            continue
        
        # Use existing global image/mask; do not copy
        global_count += 1
        base_name = os.path.splitext(os.path.basename(entry.get('img', '')))[0] or f"img_{global_count:06d}"
        img_array = np.array(img)
        h, w = mask_array.shape[:2]
        
        patch_idx = 0
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                mask_patch = mask_array[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                label_id, ratio = get_majority_class(mask_patch, valid_labels)
                
                if label_id is not None:
                    patch_count += 1
                    patch_idx += 1
                    label_stats[label_id] += 1
                    
                    img_patch = img_array[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    patch_img = Image.fromarray(img_patch)
                    
                    patch_name = f"{base_name}_{patch_idx:04d}"
                    patch_save_path = os.path.join(dirs['local_dir'], f"{patch_name}.png")
                    patch_img.save(patch_save_path)
                    
                    # Aggregate labels to single JSON later
                    if 'saved' not in dirs:
                        dirs['saved'] = []
                    dirs['saved'].append({
                        'name': f"{patch_name}.png",
                        'label': label_names[label_id],
                        'top_left': [int(x), int(y)],
                        'bottom_right': [int(x + PATCH_SIZE - 1), int(y + PATCH_SIZE - 1)],
                    })
        
        processed_images += 1
        show_progress(processed_images, total_images)

        if processed_images % 100 == 0:
            print(f"\n  Processed {processed_images}/{total_images} images, {patch_count} patches extracted")

    if total_images > 0:
        sys.stdout.write("\n")
    
    return patch_count, global_count, label_stats


def main(argv: Sequence[str] | None = None):
    print("=" * 60)
    print("RELLIS-3D Mask to Patch Processing")
    print("=" * 60)
    
    # CLI options
    import argparse
    parser = argparse.ArgumentParser(description="RELLIS-3D mask_to_patch")
    parser.add_argument("--split", choices=["train", "valid", "test"], help="Process a single split")
    parser.add_argument("--limit-image", type=int, default=None, help="Limit number of images in the split")
    parser.add_argument("--no-reset", action="store_true", help="Do not clear local output before processing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not os.path.exists(REMAINLABELS_FILE):
        print(f"Error: remainlabels.txt not found at {REMAINLABELS_FILE}")
        return
    
    label_names = load_labels(REMAINLABELS_FILE)
    valid_labels = set(label_names.keys())
    print(f"\nLoaded {len(valid_labels)} valid labels:")
    for label_id, name in sorted(label_names.items()):
        print(f"  {label_id}: {name}")
    
    splits = {
        'train': 'train',
        'valid': 'valid',
        'test': 'test'
    }
    
    total_patches = 0
    total_images = 0
    all_label_stats = defaultdict(int)
    
    target_splits = [args.split] if args.split else list(splits.keys())
    for split_name in target_splits:
        dirs = setup_directories(split_name, reset=(not args.no_reset))
        patch_count, global_count, label_stats = process_split(
            split_name, valid_labels, label_names, dirs, limit_images=args.limit_image
        )
        # Write aggregated local_label.json
        saved = dirs.get('saved', [])
        out_json = os.path.join(dirs['local_dir'], 'local_label.json')
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({'items': saved, 'count': len(saved)}, f, ensure_ascii=False, indent=2)
        
        total_patches += patch_count
        total_images += global_count
        for label_id, count in label_stats.items():
            all_label_stats[label_id] += count
        
        print(f"\n{split_name} summary:")
        print(f"  Images processed: {global_count}")
        print(f"  Patches extracted: {patch_count}")
        print(f"  Labels found: {len(label_stats)}")
        for label_id, count in sorted(label_stats.items()):
            print(f"    {label_id} ({label_names[label_id]}): {count} patches")
    
    print("\n" + "=" * 60)
    print("Overall Summary")
    print("=" * 60)
    print(f"Total images processed: {total_images}")
    print(f"Total patches extracted: {total_patches}")
    print(f"Labels found: {len(all_label_stats)}")
    for label_id, count in sorted(all_label_stats.items()):
        print(f"  {label_id} ({label_names[label_id]}): {count} patches")


def run(extra_args: Sequence[str] | None = None) -> None:
    main(extra_args)


if __name__ == "__main__":
    main()
