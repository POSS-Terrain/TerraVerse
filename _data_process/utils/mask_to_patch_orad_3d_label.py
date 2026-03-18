#!/usr/bin/env python3
import argparse
import colorsys
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

_DATASET_NAME = 'ORAD-3D-Label'


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


ID_LABEL_MAP: Dict[int, str] = {
	7: "road",
	8: "car",
	9: "grass-on-road",
	10: "people",
	11: "safe-road",
	12: "water",
	13: "snow",
	14: "rock",
	255: "road",
}


def color_for_id(label_id: int) -> Tuple[int, int, int]:
	h = (label_id * 0.61803398875) % 1.0
	s = 0.85
	v = 0.95
	r, g, b = colorsys.hsv_to_rgb(h, s, v)
	return int(r * 255), int(g * 255), int(b * 255)


def packed_color_for_id(label_id: int) -> int:
	r, g, b = color_for_id(label_id)
	return (r << 16) + (g << 8) + b


def label_for_id(majority_id: int) -> str:
	return ID_LABEL_MAP.get(majority_id, f"id_{majority_id}")


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


def packed_colors_to_ids(mask_packed: np.ndarray, color_id_map: Dict[int, int]) -> np.ndarray:
	ids = np.zeros_like(mask_packed, dtype=np.uint16)
	if mask_packed.size == 0:
		return ids
	unique_vals = np.unique(mask_packed)
	for val in unique_vals:
		label_id = color_id_map.get(int(val), 0)
		ids[mask_packed == val] = np.uint16(label_id)
	return ids


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
	mask_type: str = "auto",
) -> Tuple[int, int]:
	global_dir = split_root / "global_image"
	out_dir = out_root / "local_image"
	if out_dir.exists() and reset:
		shutil.rmtree(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	list_path = global_dir / "global_list.json"
	items = json.loads(list_path.read_text(encoding="utf-8")).get("items", [])

	results: List[PatchInfo] = []

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

		if mask_type == "auto":
			mode = "color" if mask.ndim == 3 else "id"
		elif mask_type == "labelColor":
			mode = "color"
		else:
			mode = "id"

		if mode == "color":
			packed = color_to_packed(mask)
			color_id_map = {packed_color_for_id(i): i for i in ID_LABEL_MAP.keys()}
			mask_ids = packed_colors_to_ids(packed, color_id_map)
		else:
			mask_ids = mask if mask.ndim == 2 else mask[..., 0]

		patch_idx = 0
		for r, c in iter_patches(img, patch=patch_size, stride=stride):
			img_patch = img[r : r + patch_size, c : c + patch_size]
			mask_ids_patch = mask_ids[r : r + patch_size, c : c + patch_size]

			majority_id, ratio = majority_ratio(mask_ids_patch)
			if ratio < min_majority:
				continue
			if int(majority_id) == 0:
				continue

			label_name = label_for_id(int(majority_id))

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

	label_types = {it.get("label") for it in label_items if isinstance(it, dict) and it.get("label")}
	return len(results), len(label_types)


def main(argv: Sequence[str] | None = None):
	parser = argparse.ArgumentParser(description="ORAD-3D mask_to_patch processor")
	parser.add_argument(
		"--processed-root",
		type=Path,
		default=(_default_processed_root("processed_data_1")),
		help="Path to processed_data_1 root (contains train/valid/test)",
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
	parser.add_argument(
		"--mask-type",
		type=str,
		default="auto",
		help="Mask type: auto | labelIds | labelColor",
	)
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
