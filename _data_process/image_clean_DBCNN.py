from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import pyiqa
from tqdm import tqdm

from image_clean_common import (
        load_local_label,
        reindex_local_images,
        save_local_label,
        save_records_to_csv,
        save_score_histogram,
)


def get_image_quality_scorer(model_name: str):
	print(f"Loading {model_name} model...")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	try:
		metric = pyiqa.create_metric(model_name, device=device)
	except Exception as e:
		print(f"Error loading model: {e}")
		raise
	return metric, device


def find_local_image_select_dirs(root: Path, datasets: Sequence[str] | None = None) -> List[Path]:
        """Find all local_image_select directories under either direct or legacy layout."""
        dirs: List[Path] = []
        dataset_filter = set(datasets) if datasets else None
        for p in root.rglob("local_image_select"):
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


def dataset_and_split_from_dir(local_dir: Path, root: Path) -> Tuple[str, str]:
        info = _parse_layout_dir(root, local_dir)
        if info is None:
                raise ValueError(f"Unsupported data layout: {local_dir}")
        dataset, split, _ = info
        return dataset, split


def safe_name(name: str) -> str:
	return name.replace("/", "__").replace("\\", "__")


def filter_local_image_select(
	root: Path,
	datasets: Sequence[str] | None = None,
	model_name: str = "dbcnn",
	quality_threshold: float = 0.33,
	thresholds: Dict[str, float] | None = None,
	do_delete: bool = True,
	reindex: bool = True,
	save_csv: bool = True,
	save_histogram: bool = False,
	hist_bins: int = 50,
) -> Dict[str, Dict[str, int]]:
	"""
	使用 TerraData_clean 相同的图像质量评分方法，
	筛选各数据集 processed_data/<split>/local_image_select 下的图像。

	返回统计信息: {"dataset/split": {"total": int, "kept": int, "deleted": int}}
	"""
	img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
	local_dirs = find_local_image_select_dirs(root, datasets=datasets)
	if not local_dirs:
		print("[INFO] No local_image_select directories found.")
		return {}

	scorer, _ = get_image_quality_scorer(model_name)
	stats: Dict[str, Dict[str, int]] = {}

	for local_dir in local_dirs:
		dataset, split = dataset_and_split_from_dir(local_dir, root)
		key = f"{dataset}/{split}"

		threshold = quality_threshold
		if thresholds and dataset in thresholds:
			threshold = thresholds[dataset]

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
				score = scorer(str(img_path)).item()
				scores.append(float(score))
				record = {
					"name": img_path.name,
					"score": f"{score:.4f}",
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

		# Delete files and update local_label.json
		if do_delete and deleted_names:
			for img_path in image_paths:
				if img_path.name in deleted_names:
					try:
						os.remove(img_path)
					except Exception as e:
						print(f"[WARN] Failed to delete {img_path}: {e}")

			label_path = local_dir / "local_label.json"
			label_data = load_local_label(label_path)
			items = label_data.get("items", [])
			if isinstance(items, list):
				new_items = [item for item in items if item.get("name") not in deleted_names]
				if len(new_items) != len(items):
					label_data["items"] = new_items
					save_local_label(label_path, label_data)
					if reindex:
						reindex_local_images(local_dir, label_data)
						save_local_label(label_path, label_data)

		if save_csv:
			save_records_to_csv(local_dir / "quality_kept.csv", kept_records)
			save_records_to_csv(local_dir / "quality_deleted.csv", deleted_records)

		if save_histogram:
			hist_dir = root / "Z_image_score"
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

