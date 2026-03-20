from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from image_clean_common import (
        load_local_label,
        reindex_local_images,
        save_local_label,
        save_records_to_csv,
        save_score_histogram,
)


def _parse_layout_dir(root: Path, dir_path: Path) -> Tuple[str, str, Path] | None:
        try:
                rel = dir_path.relative_to(root)
        except ValueError:
                return None

        parts = list(rel.parts)
        if len(parts) < 2:
                return None

        prefix = parts[:-1]
        if len(prefix) == 1:
                dataset = prefix[0]
                split = "all"
        elif len(prefix) == 2 and prefix[1] != "processed_data":
                dataset, split = prefix
        elif len(prefix) == 2 and prefix[1] == "processed_data":
                dataset = prefix[0]
                split = "all"
        elif len(prefix) == 3 and prefix[1] == "processed_data":
                dataset = prefix[0]
                split = prefix[2]
        else:
                return None

        base_dir = root / dataset if split == "all" else root / dataset / split
        return dataset, split, base_dir


def find_local_image_select_dirs(
        root: Path,
        datasets: Sequence[str] | None = None,
        dir_name: str = "local_image_select",
) -> List[Path]:
        """Find all stage directories under either direct or legacy layout."""
        dirs: List[Path] = []
        dataset_filter = set(datasets) if datasets else None
        for p in root.rglob(dir_name):
                if not p.is_dir():
                        continue
                info = _parse_layout_dir(root, p)
                if info is None:
                        continue
                dataset, _, _ = info
                if dataset_filter and dataset not in dataset_filter:
                        continue
                dirs.append(p)
        return sorted(dirs)


def dataset_and_split_from_dir(local_dir: Path, root: Path) -> Tuple[str, str]:
        info = _parse_layout_dir(root, local_dir)
        if info is None:
                raise ValueError(f"Unsupported data layout: {local_dir}")
        dataset, split, _ = info
        return dataset, split


def base_dir_from_dir(local_dir: Path, root: Path) -> Path:
        info = _parse_layout_dir(root, local_dir)
        if info is None:
                raise ValueError(f"Unsupported data layout: {local_dir}")
        return info[2]


def safe_name(name: str) -> str:
	return name.replace("/", "__").replace("\\", "__")


def _to_numpy_rgb(img_path: Path) -> np.ndarray:
	img = Image.open(img_path).convert("RGB")
	arr = np.asarray(img).astype(np.float32) / 255.0
	return arr


def _brightness_saturation(arr: np.ndarray) -> Tuple[float, float]:
	# HSV conversion using PIL for reliability
	img = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
	hsv = img.convert("HSV")
	hsv_arr = np.asarray(hsv).astype(np.float32) / 255.0
	# HSV: H, S, V in [0,1]
	saturation = float(hsv_arr[..., 1].mean())
	brightness = float(hsv_arr[..., 2].mean())
	return brightness, saturation


def _grayscale(arr: np.ndarray) -> np.ndarray:
	# ITU-R BT.601 luma
	return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _variance(gray: np.ndarray) -> float:
	return float(np.var(gray))


def _sharpness(gray: np.ndarray) -> float:
	# Laplacian via convolution kernel
	kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
	pad = np.pad(gray, 1, mode="edge")
	out = (
		kernel[0, 0] * pad[:-2, :-2]
		+ kernel[0, 1] * pad[:-2, 1:-1]
		+ kernel[0, 2] * pad[:-2, 2:]
		+ kernel[1, 0] * pad[1:-1, :-2]
		+ kernel[1, 1] * pad[1:-1, 1:-1]
		+ kernel[1, 2] * pad[1:-1, 2:]
		+ kernel[2, 0] * pad[2:, :-2]
		+ kernel[2, 1] * pad[2:, 1:-1]
		+ kernel[2, 2] * pad[2:, 2:]
	)
	return float(np.var(out))


def _entropy(gray: np.ndarray) -> float:
	# 8-bit histogram entropy
	vals = (gray * 255).clip(0, 255).astype(np.uint8).ravel()
	hist = np.bincount(vals, minlength=256).astype(np.float64)
	prob = hist / np.sum(hist)
	prob = prob[prob > 0]
	return float(-np.sum(prob * np.log2(prob)))


def compute_metrics(img_path: Path) -> Dict[str, float]:
	arr = _to_numpy_rgb(img_path)
	brightness, saturation = _brightness_saturation(arr)
	gray = _grayscale(arr)
	variance = _variance(gray)
	sharpness = _sharpness(gray)
	entropy = _entropy(gray)
	return {
		"brightness": brightness,
		"saturation": saturation,
		"variance": variance,
		"sharpness": sharpness,
		"entropy": entropy,
	}


def score_weighted(
	metrics: Dict[str, float],
	brightness_target: float,
	brightness_range: float,
	sat_norm: float,
	var_norm: float,
	sharpness_norm: float,
	entropy_norm: float,
	weights: Dict[str, float],
) -> float:
	brightness = metrics["brightness"]
	brightness_score = 1.0 - min(abs(brightness - brightness_target) / max(brightness_range, 1e-6), 1.0)
	saturation_score = min(metrics["saturation"] / max(sat_norm, 1e-6), 1.0)
	variance_score = min(metrics["variance"] / max(var_norm, 1e-6), 1.0)
	sharpness_score = min(metrics["sharpness"] / max(sharpness_norm, 1e-6), 1.0)
	entropy_score = min(metrics["entropy"] / max(entropy_norm, 1e-6), 1.0)

	score = (
		weights["brightness"] * brightness_score
		+ weights["saturation"] * saturation_score
		+ weights["variance"] * variance_score
		+ weights["sharpness"] * sharpness_score
		+ weights["entropy"] * entropy_score
	)
	return float(score)


def filter_local_image_select(
	root: Path,
	datasets: Sequence[str] | None = None,
	quality_threshold: float = 0.5,
	do_delete: bool = True,
	reindex: bool = True,
	save_csv: bool = True,
	save_histogram: bool = False,
	hist_bins: int = 50,
	input_dir_name: str = "local_image_select_2",
	output_dir_name: str = "local_image_select_3",
	brightness_target: float = 0.5,
	brightness_range: float = 0.35,
	sat_norm: float = 0.3,
	var_norm: float = 0.02,
	sharpness_norm: float = 0.02,
	entropy_norm: float = 7.5,
	weights: Dict[str, float] | None = None,
) -> Dict[str, Dict[str, int]]:
	"""
	使用简单统计指标筛选图像：亮度、饱和度、灰度方差、清晰度(拉普拉斯方差)、熵。
	仅使用加权得分，得分 >= quality_threshold 视为保留。

	返回统计信息: {"dataset/split": {"total": int, "kept": int, "deleted": int}}
	"""
	img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
	local_dirs = find_local_image_select_dirs(root, datasets=datasets, dir_name=input_dir_name)
	if not local_dirs:
		print(f"[INFO] No {input_dir_name} directories found.")
		return {}

	if weights is None:
		weights = {
			"brightness": 0.25,
			"saturation": 0.15,
			"variance": 0.25,
			"sharpness": 0.25,
			"entropy": 0.10,
		}

	stats: Dict[str, Dict[str, int]] = {}

	for local_dir in local_dirs:
		dataset, split = dataset_and_split_from_dir(local_dir, root)
		key = f"{dataset}/{split}"

		threshold = quality_threshold

		image_paths = sorted(
			[p for p in local_dir.iterdir() if p.suffix.lower() in img_exts]
		)
		total = len(image_paths)
		if total == 0:
			stats[key] = {"total": 0, "kept": 0, "deleted": 0}
			continue

		kept_records: List[Dict[str, str]] = []
		deleted_records: List[Dict[str, str]] = []
		deleted_names = set()
		scores: List[float] = []

		for img_path in tqdm(image_paths, desc=f"{key}"):
			try:
				metrics = compute_metrics(img_path)
				score = score_weighted(
					metrics,
					brightness_target=brightness_target,
					brightness_range=brightness_range,
					sat_norm=sat_norm,
					var_norm=var_norm,
					sharpness_norm=sharpness_norm,
					entropy_norm=entropy_norm,
					weights=weights,
				)
				scores.append(float(score))
				record = {
					"name": img_path.name,
					"score": f"{score:.4f}",
					"brightness": f"{metrics['brightness']:.4f}",
					"saturation": f"{metrics['saturation']:.4f}",
					"variance": f"{metrics['variance']:.6f}",
					"sharpness": f"{metrics['sharpness']:.6f}",
					"entropy": f"{metrics['entropy']:.4f}",
					"full_path": str(img_path),
					"split": split,
					"dataset": dataset,
				}

				if score < threshold:
					deleted_records.append(record)
					deleted_names.add(img_path.name)
				else:
					kept_records.append(record)
			except Exception as e:
				print(f"[WARN] Skip {img_path}: {e}")

		# Save to output dir without modifying source folder
		output_dir = None
		if do_delete:
			output_dir = base_dir_from_dir(local_dir, root) / output_dir_name
			if output_dir.exists():
				for p in output_dir.iterdir():
					if p.is_file():
						p.unlink()
					elif p.is_dir():
						for sub in p.rglob("*"):
							if sub.is_file():
								sub.unlink()
						p.rmdir()
			output_dir.mkdir(parents=True, exist_ok=True)
			for img_path in image_paths:
				if img_path.name not in deleted_names:
					try:
						from shutil import copy2

						copy2(img_path, output_dir / img_path.name)
					except Exception as e:
						print(f"[WARN] Failed to copy {img_path}: {e}")

			label_path = local_dir / "local_label.json"
			label_data = load_local_label(label_path)
			items = label_data.get("items", [])
			if isinstance(items, list):
				new_items = [item for item in items if item.get("name") not in deleted_names]
				label_data["items"] = new_items
				save_local_label(output_dir / "local_label.json", label_data)
				if reindex:
					reindex_local_images(output_dir, label_data)
					save_local_label(output_dir / "local_label.json", label_data)

		if save_csv:
			csv_dir = output_dir if output_dir is not None else local_dir
			save_records_to_csv(csv_dir / "quality_kept.csv", kept_records)
			save_records_to_csv(csv_dir / "quality_deleted.csv", deleted_records)

		if save_histogram:
			hist_dir = root / "L_image_score"
			hist_dir.mkdir(parents=True, exist_ok=True)
			name = safe_name(f"{dataset}_{split}")
			hist_path = hist_dir / f"{name}_quality_score_hist.png"
			save_score_histogram(scores, hist_path, bins=hist_bins)

		stats[key] = {
			"total": total,
			"kept": len(kept_records),
			"deleted": len(deleted_records),
		}

	return stats

