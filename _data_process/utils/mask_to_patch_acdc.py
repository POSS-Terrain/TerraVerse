#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
from PIL import Image



try:
    from ._dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root
except ImportError:
    from _dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root

_DATASET_NAME = 'ACDC'


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

@dataclass
class PatchInfo:
	src_img: str
	src_mask: str
	patch_img: str
	top: int
	left: int
	row_idx: int
	col_idx: int


CITYSCAPES_LABELS = [
	{"name": "road", "id": 7, "trainId": 0, "color": (128, 64, 128)},
	{"name": "sidewalk", "id": 8, "trainId": 1, "color": (244, 35, 232)},
	{"name": "building", "id": 11, "trainId": 2, "color": (70, 70, 70)},
	{"name": "wall", "id": 12, "trainId": 3, "color": (102, 102, 156)},
	{"name": "fence", "id": 13, "trainId": 4, "color": (190, 153, 153)},
	{"name": "pole", "id": 17, "trainId": 5, "color": (153, 153, 153)},
	{"name": "traffic light", "id": 19, "trainId": 6, "color": (250, 170, 30)},
	{"name": "traffic sign", "id": 20, "trainId": 7, "color": (220, 220, 0)},
	{"name": "vegetation", "id": 21, "trainId": 8, "color": (107, 142, 35)},
	{"name": "terrain", "id": 22, "trainId": 9, "color": (152, 251, 152)},
	{"name": "sky", "id": 23, "trainId": 10, "color": (70, 130, 180)},
	{"name": "person", "id": 24, "trainId": 11, "color": (220, 20, 60)},
	{"name": "rider", "id": 25, "trainId": 12, "color": (255, 0, 0)},
	{"name": "car", "id": 26, "trainId": 13, "color": (0, 0, 142)},
	{"name": "truck", "id": 27, "trainId": 14, "color": (0, 0, 70)},
	{"name": "bus", "id": 28, "trainId": 15, "color": (0, 60, 100)},
	{"name": "train", "id": 31, "trainId": 16, "color": (0, 80, 100)},
	{"name": "motorcycle", "id": 32, "trainId": 17, "color": (0, 0, 230)},
	{"name": "bicycle", "id": 33, "trainId": 18, "color": (119, 11, 32)},
]


def build_label_maps() -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
	id_to_name = {item["id"]: item["name"] for item in CITYSCAPES_LABELS}
	train_id_to_name = {item["trainId"]: item["name"] for item in CITYSCAPES_LABELS}
	color_to_name = {}
	for item in CITYSCAPES_LABELS:
		r, g, b = item["color"]
		packed = (int(r) << 16) + (int(g) << 8) + int(b)
		color_to_name[packed] = item["name"]
	return id_to_name, train_id_to_name, color_to_name


def load_png(path: Path) -> np.ndarray:
	return np.array(Image.open(path))


def save_png(path: Path, array: np.ndarray) -> None:
	Image.fromarray(array).save(path)


def color_to_packed(mask_rgb: np.ndarray) -> np.ndarray:
	if mask_rgb.ndim == 2:
		return mask_rgb
	r = mask_rgb[..., 0].astype(np.uint32)
	g = mask_rgb[..., 1].astype(np.uint32)
	b = mask_rgb[..., 2].astype(np.uint32)
	return (r << 16) + (g << 8) + b


def iter_patches(img: np.ndarray, patch: int = 224, stride: Optional[int] = None):
	stride = (patch // 2) if stride is None else stride
	h, w = img.shape[:2]
	for top in range(0, h - patch + 1, stride):
		for left in range(0, w - patch + 1, stride):
			yield top, left


def majority_ratio(mask_ids_patch: np.ndarray) -> Tuple[int, float]:
	vals, counts = np.unique(mask_ids_patch.reshape(-1), return_counts=True)
	idx = int(np.argmax(counts))
	majority_id = int(vals[idx])
	ratio = float(counts[idx]) / float(mask_ids_patch.size)
	return majority_id, ratio


def process_split(
	split_root: Path,
	out_root: Path,
	patch_size: int = 224,
	stride: Optional[int] = None,
	min_majority: float = 0.95,
	limit_images: Optional[int] = None,
	reset: bool = True,
	mask_type: str = "labelIds",
) -> Tuple[int, int]:
	global_dir = split_root / "global_image"
	out_dir = out_root / "local_image"
	if out_dir.exists() and reset:
		shutil.rmtree(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	list_path = global_dir / "global_list.json"
	items = json.loads(list_path.read_text(encoding="utf-8")).get("items", [])

	results: List[PatchInfo] = []
	id_to_name, train_id_to_name, color_to_name = build_label_maps()
	if mask_type == "labelTrainIds":
		label_map = train_id_to_name
	elif mask_type == "labelColor":
		label_map = color_to_name
	else:
		label_map = id_to_name

	label_items: List[dict] = []

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

	for i, it in enumerate(items, start=1):
		if limit_images and i > limit_images:
			break
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

		img = load_png(img_path)
		mask = load_png(mask_path)

		if mask_type == "labelColor":
			mask_ids = color_to_packed(mask)
		else:
			mask_ids = mask if mask.ndim == 2 else mask[..., 0]

		patch_idx = 0
		for r, c in iter_patches(img, patch=patch_size, stride=stride):
			img_patch = img[r : r + patch_size, c : c + patch_size]
			mask_ids_patch = mask_ids[r : r + patch_size, c : c + patch_size]

			majority_id, ratio = majority_ratio(mask_ids_patch)
			if ratio < min_majority:
				continue

			label_name = label_map.get(int(majority_id))
			if label_name is None:
				continue

			patch_idx += 1
			base = Path(img_name).stem
			out_img_name = f"{base}_{patch_idx:04d}.png"
			save_png(out_dir / out_img_name, img_patch)

			label_items.append(
				{
					"name": out_img_name,
					"label": label_name,
					"top_left": [int(c), int(r)],
					"bottom_right": [int(c + patch_size - 1), int(r + patch_size - 1)],
				}
			)

			results.append(
				PatchInfo(
					src_img=img_name,
					src_mask=mask_name,
					patch_img=out_img_name,
					top=r,
					left=c,
					row_idx=r // patch_size,
					col_idx=c // patch_size,
				)
			)

		processed_images += 1
		show_progress(processed_images, total_images)

	if total_images > 0:
		sys.stdout.write("\n")

	local_label = {"items": label_items}
	(out_dir / "local_label.json").write_text(
		json.dumps(local_label, ensure_ascii=False, indent=2), encoding="utf-8"
	)

	return len(results), len(set(label_map.values()))


def main(argv: Sequence[str] | None = None):
	parser = argparse.ArgumentParser(description="ACDC mask_to_patch processor")
	parser.add_argument(
		"--processed-root",
		type=Path,
		default=(_default_processed_root()),
		help="Path to processed_data root (contains train/valid/test)",
	)
	parser.add_argument("--split", type=str, help="Process a single split (train/valid/test)")
	parser.add_argument(
		"--splits",
		nargs="*",
		default=["train", "valid", "test"],
		help="Which splits to process",
	)
	parser.add_argument("--patch-size", type=int, default=224)
	parser.add_argument("--stride", type=int, default=None, help="Stride for sliding window; default=half of patch-size")
	parser.add_argument("--min-majority", type=float, default=0.95)
	parser.add_argument("--limit-image", type=int, default=None, help="Limit number of source images per split (single form)")
	parser.add_argument("--limit-images", type=int, default=None, help="Limit number of source images per split (plural form)")
	parser.add_argument("--mask-type", type=str, default="labelIds", help="Mask type: labelIds | labelTrainIds | labelColor")
	parser.add_argument("--no-reset", action="store_true", help="Do not clear existing local_image contents")
	args = parser.parse_args(list(argv) if argv is not None else None)

	split_names = [args.split] if args.split else args.splits
	limit_images = args.limit_image if args.limit_image is not None else args.limit_images

	for split in split_names:
		split_root = args.processed_root / split
		if not (split_root / "global_image" / "global_list.json").exists():
			continue
		out_root = split_root
		print(f"Processing split: {split}")
		n_patches, n_labels = process_split(
			split_root=split_root,
			out_root=out_root,
			patch_size=args.patch_size,
			stride=args.stride,
			min_majority=args.min_majority,
			limit_images=limit_images,
			reset=(not args.no_reset),
			mask_type=args.mask_type,
		)
		print(f"  -> accepted patches: {n_patches}, labels: {n_labels} (output: {split_root / 'local_image'})")


def run(extra_args: Sequence[str] | None = None) -> None:
    main(extra_args)


if __name__ == "__main__":
    main()
