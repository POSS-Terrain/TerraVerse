"""
RTK: 读取 processed_data/train/global_image 下的701张图像，结合对应 Json 标注切割 patch
输出到 processed_data/train/local_image，并生成 local_label.json
两种patch_size
"""
import argparse
import json
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

_DATASET_NAME = 'RTK'


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
ratio_confidence = 0.85

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
def list_images(folder: Path) -> List[Path]:
	return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])

def ensure_clean_dir(path: Path, reset: bool) -> None:
	if reset and path.exists():
		shutil.rmtree(path)
	path.mkdir(parents=True, exist_ok=True)


def iter_patches_from_bottom_left(
	height: int,
	width: int,
	patch_size: int,
	stride: int,
):
	start_y = height - patch_size
	for top in range(start_y, -1, -stride):
		for left in range(0, width - patch_size + 1, stride):
			yield top, left


def load_labelme_mask(
	json_path: Path,
	image_size: Tuple[int, int],
	label_to_id: Dict[str, int],
) -> np.ndarray:
	with json_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	width, height = image_size
	mask = np.zeros((height, width), dtype=np.uint16)

	for shape in data.get("shapes", []):
		label = str(shape.get("label", "unknown")).strip()
		points = shape.get("points", [])
		shape_type = shape.get("shape_type", "polygon")
		if not label or shape_type != "polygon" or len(points) < 3:
			continue

		if label not in label_to_id:
			label_to_id[label] = len(label_to_id) + 1
		label_id = label_to_id[label]

		poly = [(float(x), float(y)) for x, y in points]
		poly_mask = Image.new("1", (width, height), 0)
		draw = ImageDraw.Draw(poly_mask)
		draw.polygon(poly, outline=1, fill=1)
		poly_arr = np.array(poly_mask, dtype=bool)
		mask[poly_arr] = label_id

	return mask


def majority_label(mask_patch: np.ndarray, threshold_ratio: float) -> Tuple[Optional[int], float]:
	values, counts = np.unique(mask_patch, return_counts=True)
	total = mask_patch.size
	best_id = None
	best_ratio = 0.0

	for val, cnt in zip(values, counts):
		if int(val) == 0:
			continue
		ratio = cnt / total
		if ratio > best_ratio:
			best_ratio = ratio
			best_id = int(val)

	if best_id is None or best_ratio < threshold_ratio:
		return None, 0.0
	return best_id, best_ratio


def process(
	data_root: Path,
	global_dir: Path,
	output_dir: Path,
	json_dir: Path,
	patch_sizes: List[int],
	threshold_ratio: float,
	reset: bool,
	limit_images: Optional[int],
) -> None:
	ensure_clean_dir(output_dir, reset=reset)

	images = list_images(global_dir)
	if not images:
		raise FileNotFoundError(f"No global_image files found in: {global_dir}")

	label_to_id: Dict[str, int] = {}
	id_to_label: Dict[int, str] = {}
	label_items = []
	total_images = len(images) if limit_images is None else min(len(images), limit_images)

	def show_progress(done: int, total: int, current_name: str) -> None:
		remaining = max(total - done, 0)
		bar_width = 30
		ratio = min(max(done / max(total, 1), 0.0), 1.0)
		filled = int(bar_width * ratio)
		bar = "#" * filled + "-" * (bar_width - filled)
		percent = int(ratio * 100)
		print(
			f"\rProgress [{bar}] {percent:3d}% | Processed {done}/{total} | Remaining {remaining} | Current {current_name}",
			end="",
			flush=True,
		)

	processed = 0
	iter_images = images[:total_images]
	if tqdm is not None:
		iter_images = tqdm(iter_images, total=total_images, desc="processing")

	for idx, img_path in enumerate(iter_images, start=1):
		img_name = img_path.name
		try:
			img_id = int(img_name.replace("img_", "").split(".")[0])
		except ValueError:
			print(f"Skipping file with unparsable ID: {img_name}")
			if tqdm is None:
				processed += 1
				show_progress(processed, total_images, img_name)
			continue

		json_id = img_id - 1
		json_path = json_dir / f"{json_id:09d}.json"
		if not json_path.exists():
			print(f"Missing JSON: {json_path.name}, skipping {img_name}")
			if tqdm is None:
				processed += 1
				show_progress(processed, total_images, img_name)
			continue

		img = Image.open(img_path)
		width, height = img.size
		mask = load_labelme_mask(json_path, (width, height), label_to_id)
		id_to_label = {v: k for k, v in label_to_id.items()}

		patch_count = 0
		# 统一阈值到 ratio_confidence
		threshold_ratio = ratio_confidence
		for patch_size in patch_sizes:
            # 步长调整
			stride = patch_size // 2
			for top, left in iter_patches_from_bottom_left(height, width, patch_size, stride):
				bottom = top + patch_size
				right = left + patch_size
				if bottom > height or right > width:
					continue

				mask_patch = mask[top:bottom, left:right]
				label_id, ratio = majority_label(mask_patch, threshold_ratio)
				if label_id is None:
					continue

				patch_count += 1
				patch_name = f"{img_path.stem}_{patch_count:04d}.png"
				patch_img = img.crop((left, top, right, bottom))
				if patch_size == 112:
					patch_img = patch_img.resize((224, 224), Image.LANCZOS)
				patch_img.save(output_dir / patch_name)

				if ratio < ratio_confidence:
					continue

				label_items.append(
					{
						"name": patch_name,
						"label": id_to_label.get(label_id, f"unknown_{label_id}"),
						"top_left": [int(left), int(top)],
						"bottom_right": [int(right - 1), int(bottom - 1)],
						"patch_size": int(patch_size),
					}
				)

		if tqdm is None:
			processed += 1
			show_progress(processed, total_images, img_name)

	if tqdm is None and total_images > 0:
		print("")

	with (output_dir / "local_label.json").open("w", encoding="utf-8") as f:
		json.dump({"items": label_items}, f, ensure_ascii=False, indent=2)
		f.write("\n")

	print(f"Total patch count: {len(label_items)}")


def main(argv: Sequence[str] | None = None):
	parser = argparse.ArgumentParser(description="RTK mask-to-patch")
	parser.add_argument(
		"--data-root",
		type=Path,
		default=_default_dataset_root(),
		help="RTK dataset root directory",
	)
	parser.add_argument(
		"--global-dir",
		type=Path,
		default=None,
		help="global_image directory. Defaults to <data-root>/global_image",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Output local_image directory. Defaults to <data-root>/local_image",
	)
	parser.add_argument(
		"--json-dir",
		type=Path,
		default=None,
		help="JSON annotation directory. Defaults to <data-root>/RTK_SemanticSegmentationGT_Json",
	)
	parser.add_argument("--patch-sizes", nargs="*", type=int, default=[112, 224], help="List of patch sizes")
	parser.add_argument("--threshold-ratio", type=float, default=ratio_confidence, help="Threshold ratio for keeping a patch")
	parser.add_argument("--limit-images", type=int, default=None, help="Limit the number of images to process")
	parser.add_argument("--no-reset", action="store_true", help="Do not clear the output directory")
	args = parser.parse_args(list(argv) if argv is not None else None)

	base_dir = args.data_root if (args.data_root / "global_image").exists() else (args.data_root if _looks_like_processed_root(args.data_root) else args.data_root / "processed_data")
	global_dir = args.global_dir or (base_dir / "global_image")
	output_dir = args.output_dir or (base_dir / "local_image")
	json_dir = args.json_dir or (args.data_root / "RTK_SemanticSegmentationGT_Json")

	process(
		data_root=args.data_root,
		global_dir=global_dir,
		output_dir=output_dir,
		json_dir=json_dir,
		patch_sizes=args.patch_sizes,
		threshold_ratio=args.threshold_ratio,
		reset=(not args.no_reset),
		limit_images=702
	)


def test_run():
	data_root = Path(__file__).resolve().parent.parent
	process(
		data_root=data_root,
		global_dir=data_root / "global_image",
		output_dir=data_root / "local_image",
		json_dir=data_root / "RTK_SemanticSegmentationGT_Json",
		patch_sizes=[112, 224],
		threshold_ratio=0.4,
		reset=True,
		limit_images=15,
	)


if __name__ == "__main__":
	#test_run()
	main()


def run(extra_args: Sequence[str] | None = None) -> None:
    main(extra_args)
