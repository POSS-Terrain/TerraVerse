"""
* GT_color:	   folder containing the groundtruth masks for semantic segmentation 
                   Annotations are given using a color representation, where each color corresponds to a
                   specific class. This is primairly provided for visualization. For training, create a
                   corresponding ID image by assigning the colors to a speific class ID as given below

Class		R	G	B	ID


Void		- 	- 	-	0

Road            170 	170 	170	1

Grass           0 	255 	0	2

Vegetation      102 	102 	51	3

Tree            0 	60 	0	3

Sky             0 	120 	255	4

Obstacle        0 	0 	0	5 
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm


try:
    from ._dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root
except ImportError:
    from _dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root

_DATASET_NAME = 'DeepScene'


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

MODE = "test"  
CONFIDENCE_THRESHOLD = 0.85
PATCH_SIZE = 224
STRIDE = 112
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COLOR_TO_CLASS: Dict[Tuple[int, int, int], int] = {
    (170, 170, 170): 1,  # Road
    (0, 255, 0): 2,      # Grass
    (102, 102, 51): 3,   # Vegetation
    (0, 60, 0): 3,       # Tree
    (0, 120, 255): 4,    # Sky
    (0, 0, 0): 5,        # Obstacle
}
CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "void",
    1: "road",
    2: "grass",
    3: "vegetation",
    4: "sky",
    5: "obstacle",
}

def list_images(folder: Path, prefix: Optional[str] = None) -> List[Path]:
    paths = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if prefix is not None:
        paths = [p for p in paths if p.name.startswith(prefix)]
    return sorted(paths)

def ensure_clean_dir(path: Path, reset: bool) -> None:
	if reset and path.exists():
		shutil.rmtree(path)
	path.mkdir(parents=True, exist_ok=True)

def parse_index_from_name(path: Path) -> int:
    name = path.stem
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid filename: {path.name}")
    return int(parts[-1])

def pair_images(img_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    imgs = list_images(img_dir, prefix="img_")
    masks = list_images(mask_dir, prefix="masked_")

    if not imgs:
        raise FileNotFoundError(f"No images found in: {img_dir}")
    if not masks:
        raise FileNotFoundError(f"No masks found in: {mask_dir}")

    mask_map = {parse_index_from_name(p): p for p in masks}
    pairs = []
    for img in imgs:
        idx = parse_index_from_name(img)
        if idx not in mask_map:
            raise FileNotFoundError(f"Mask not found for image idx {idx:06d}")
        pairs.append((img, mask_map[idx]))
    return pairs

def mask_rgb_to_class(mask_rgb: np.ndarray) -> np.ndarray:
    class_map = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for color, class_id in COLOR_TO_CLASS.items():
        match = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        class_map[match] = class_id
    return class_map

def get_patch_positions(height: int, width: int) -> List[Tuple[int, int]]:
    y_starts = list(range(height - PATCH_SIZE, -1, -STRIDE))
    x_starts = list(range(0, width - PATCH_SIZE + 1, STRIDE))
    return [(y, x) for y in y_starts for x in x_starts]

def majority_label(patch_class: np.ndarray, num_classes: int = 6) -> Tuple[int, float]:
    counts = np.bincount(patch_class.ravel(), minlength=num_classes)
    label = int(np.argmax(counts))
    confidence = float(counts[label]) / float(patch_class.size)
    return label, confidence

def process_dataset(
    img_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    reset: bool = False,
) -> Path:
    ensure_clean_dir(output_dir, reset)
    pairs = pair_images(img_dir, mask_dir)

    items = []
    label_items = []
    total_patches = 0  
    for img_path, mask_path in tqdm(pairs, desc="Processing images"):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_np = np.array(img)
        with Image.open(mask_path) as mask:
            mask = mask.convert("RGB")
            mask_np = np.array(mask)

        if img_np.shape[:2] != mask_np.shape[:2]:
            raise ValueError(f"Size mismatch: {img_path.name} vs {mask_path.name}")

        class_map = mask_rgb_to_class(mask_np)
        height, width = class_map.shape
        positions = get_patch_positions(height, width)

        img_idx = parse_index_from_name(img_path)
        valid_cnt = 0
        for (y, x) in tqdm(positions, desc=f"Processing patches for {img_path.name}", leave=False):
            patch_img = img_np[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patch_cls = class_map[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            label, confidence = majority_label(patch_cls)

            if label == 4:
                continue
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            valid_cnt += 1
            total_patches += 1  
            patch_name = f"img_{img_idx:06d}_{valid_cnt:04d}.png"
            patch_path = output_dir / patch_name
            Image.fromarray(patch_img).save(patch_path, format="PNG")
            items.append({"img": patch_name, "label": label, "confidence": confidence})
            label_items.append({
                "name": patch_name,
                "label": CLASS_ID_TO_NAME.get(label, "unknown"),
                "top_left": [int(x), int(y)],
                "bottom_right": [int(x + PATCH_SIZE - 1), int(y + PATCH_SIZE - 1)]
            })

    print(f"Total patches created: {total_patches}") 

    json_path = output_dir / "local_list.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)
    label_json_path = output_dir / "local_label.json"
    with label_json_path.open("w", encoding="utf-8") as f:
        json.dump({"items": label_items}, f, ensure_ascii=False, indent=2)
    return json_path

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cut DeepScene global images into local patches and label them.")
    base_dir = _default_processed_root()
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=base_dir / MODE / "global_image",
        help="Input image directory",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=base_dir / MODE / "global_image",
        help="Input mask directory (same directory as the images)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / MODE / "local_image",
        help="Output patch directory",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the output directory before processing",
    )
    return parser

def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    json_path = process_dataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        reset=args.reset,
    )
    print(f"Saved {json_path}")

def run(extra_args: Sequence[str] | None = None) -> None:
    main(extra_args)


if __name__ == "__main__":
    main()
