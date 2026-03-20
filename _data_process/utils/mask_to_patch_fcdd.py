import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


try:
    from ._dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root
except ImportError:
    from _dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root

_DATASET_NAME = 'FCDD'


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

# **
ratio_confidence = 0.95

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
def list_images(folder: Path) -> List[Path]:
	return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])

def ensure_clean_dir(path: Path, reset: bool) -> None:
	if reset and path.exists():
		shutil.rmtree(path)
	path.mkdir(parents=True, exist_ok=True)

def load_label_mapping(label_file: Path) -> Dict[int, str]:
    label_mapping = {}
    with open(label_file, "r") as f:
        for line in f.readlines()[1:]:  
            pixel_value, label = line.strip().split(",")
            label_mapping[int(pixel_value)] = label
    return label_mapping

def cut_patches(image: np.ndarray, patch_size: int, stride: int) -> List[np.ndarray]:
    patches = []
    h, w = image.shape[:2]
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append((patch, x, y))
    return patches

def ensure_empty_dir(path: Path, reset: bool) -> None:
    if reset and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def process(
    data_root: Path,
    output_dir: Path,
    label_mapping: Dict[int, str],
    patch_size: int = 224,
    stride: int = 112,
    confidence: float = 0.95,
    limit_images: Optional[int] = None,
):
    img_files = sorted([f for f in data_root.iterdir() if f.name.startswith("img_") and f.suffix.lower() == ".png"])
    mask_files = sorted([f for f in data_root.iterdir() if f.name.startswith("masked_") and f.suffix.lower() == ".png"])

    if limit_images is not None:
        img_files = img_files[:limit_images]
        mask_files = mask_files[:limit_images]

    if len(img_files) != len(mask_files):
        raise ValueError("Mismatched number of images and masks")

    ensure_empty_dir(output_dir, reset=True)

    items = []

    for idx, (img_file, mask_file) in enumerate(
        tqdm(zip(img_files, mask_files), total=len(img_files), desc="Processing images"),
        start=1,
    ):
        img = Image.open(img_file)
        mask = Image.open(mask_file)

        img_array = np.array(img)
        mask_array = np.array(mask)

        patches = cut_patches(img_array, patch_size, stride)
        mask_patches = cut_patches(mask_array, patch_size, stride)

        valid_cnt = 0
        for patch_idx, ((patch, x, y), (mask_patch, _, _)) in enumerate(zip(patches, mask_patches), start=1):
            unique, counts = np.unique(mask_patch, return_counts=True)
            pixel_counts = dict(zip(unique, counts))

            majority_label = max(pixel_counts, key=pixel_counts.get)
            majority_count = pixel_counts[majority_label]
            total_pixels = mask_patch.size
            majority_confidence = majority_count / total_pixels

            if majority_confidence >= confidence and majority_label != 0:
                valid_cnt += 1
                patch_name = f"img_{idx:06d}_{valid_cnt:04d}.png"
                patch_label = label_mapping.get(majority_label, "unknown")

                patch_img = Image.fromarray(patch)
                patch_img.save(output_dir / patch_name)

                items.append({
                    "name": patch_name,
                    "label": patch_label,
                    "top_left": [int(x), int(y)],
                    "bottom_right": [int(x + patch_size - 1), int(y + patch_size - 1)],
                })

    json_path = output_dir / "local_label.json"
    with (json_path).open("w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)
        f.write("\n")

def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="FCDD local_image patch processing")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_default_dataset_root(),
        help="FCDD dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <data-root>/processed_data/train/local_image",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=_default_dataset_root() / "_classes.csv",
        help="Label file path",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Limit the number of images to process",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    #output_dir = args.output_dir or (args.data_root / "processed_data" / "train" / "local_image")
    output_dir = args.output_dir or ((args.data_root if _looks_like_processed_root(args.data_root) else args.data_root / "processed_data") / "valid" / "local_image")
    label_mapping = load_label_mapping(args.label_file)

    process(
        #data_root=args.data_root / "processed_data" / "train" / "global_image",
        data_root=(args.data_root if _looks_like_processed_root(args.data_root) else args.data_root / "processed_data") / "valid" / "global_image",
        output_dir=output_dir,
        label_mapping=label_mapping,
        limit_images=args.limit_images,
    )


def test():
    data_root = Path(__file__).resolve().parent.parent
    global_dir = data_root / "processed_data" / "train" / "global_image"
    output_dir = data_root / "processed_data" / "train" / "local_image"
    label_file = _default_dataset_root() / "_classes.csv"

    label_mapping = load_label_mapping(label_file)

    img_files = sorted([f for f in global_dir.iterdir() if f.name.startswith("img_") and f.suffix.lower() == ".png"])[:5]
    mask_files = sorted([f for f in global_dir.iterdir() if f.name.startswith("masked_") and f.suffix.lower() == ".png"])[:5]

    if len(img_files) != len(mask_files):
        raise ValueError("Mismatched number of images and masks")

    ensure_empty_dir(output_dir, reset=True)

    items = []

    for idx, (img_file, mask_file) in enumerate(tqdm(zip(img_files, mask_files), total=len(img_files), desc="Processing images"), start=1):
        img = Image.open(img_file)
        mask = Image.open(mask_file)

        img_array = np.array(img)
        mask_array = np.array(mask)

        patches = cut_patches(img_array, 224, 112)
        mask_patches = cut_patches(mask_array, 224, 112)

        valid_cnt = 0
        for patch_idx, ((patch, x, y), (mask_patch, _, _)) in enumerate(zip(patches, mask_patches), start=1):
            unique, counts = np.unique(mask_patch, return_counts=True)
            pixel_counts = dict(zip(unique, counts))

            majority_label = max(pixel_counts, key=pixel_counts.get)
            majority_count = pixel_counts[majority_label]
            total_pixels = mask_patch.size
            majority_confidence = majority_count / total_pixels

            if majority_confidence >= 0.95 and majority_label != 0:
                valid_cnt += 1
                patch_name = f"img_{idx:06d}_{valid_cnt:04d}.png"
                patch_label = label_mapping.get(majority_label, "unknown")

                patch_img = Image.fromarray(patch)
                patch_img.save(output_dir / patch_name)

                items.append({
                    "name": patch_name,
                    "label": patch_label,
                    "top_left": [int(x), int(y)],
                    "bottom_right": [int(x + 224 - 1), int(y + 224 - 1)],
                })

    json_path = output_dir / "local_label.json"
    with (json_path).open("w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Test completed: Processed {len(items)} patches from 5 image-mask pairs.")

def cal():
    data_root = Path(__file__).resolve().parent.parent
    #local_dir = data_root / "processed_data" / "train" / "local_image"
    local_dir = data_root / "processed_data" / "valid" / "local_image"

    total_patches = len([p for p in local_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    print(f"Total patch count: {total_patches}")
    
if __name__ == "__main__":
    
    main()
    cal()
   # test()


def run(extra_args: Sequence[str] | None = None) -> None:
    main(extra_args)
