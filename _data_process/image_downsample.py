#!/usr/bin/env python3
"""Downsample local_image_select_4 by dataset/label keep ratios.

Rules:
- Scan processed_data/(<split>/)?local_image_select_4
- Randomly sample items according to dataset ratios and per-field label ratios
- Copy selected images into local_image_final
- Update local_label.json and reindex patch IDs (same logic as image_clean_DBCNN.py)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_LABEL_RATIOS = {
        "function": {
                "ACDC": {
                        "road": 0.8,
                },
                "IDD": {
                        "road": 0.8,
                },
                "KITTI-360": {
                        "road": 0.8,
                },
                "ORAD-3D-Label": {
                        "road": 0.3,
                },
        },
        "material": {
                "RELLIS": {
                        "grass": 0.5,
                },
                "RUGD": {
                        "grass": 0.8,
                },
                "WildScenes": {
                        "dirt": 0.8,
                        "grass": 0.8,
                },
        },
        "traversability": {
                "ORAD-3D": {
                        "non-traversable": 0.5,
                },
                "ORFD": {
                        "non-traversable": 0.3,
                },
        },
}


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


def find_local_image_select_dirs(root: Path, datasets: Sequence[str] | None = None) -> List[Path]:
        """Find all local_image_select_4 directories under either direct or legacy layout."""
        dirs: List[Path] = []
        dataset_filter = set(datasets) if datasets else None
        for p in root.rglob("local_image_select_4"):
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


def load_local_label(path: Path) -> Dict:
	if not path.exists():
		return {"items": []}
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def save_local_label(path: Path, data: Dict) -> None:
	with path.open("w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)


def reindex_local_images(local_dir: Path, label_data: Dict) -> None:
	"""Reindex patch indices within each base id, keep base id unchanged."""
	items = label_data.get("items", [])
	if not isinstance(items, list) or not items:
		return

	# Group by base id (e.g., img_000001)
	groups: Dict[str, List[Dict]] = defaultdict(list)
	for item in items:
		name = item.get("name")
		if not name:
			continue
		stem = Path(name).stem
		if "_" in stem:
			base = stem.rsplit("_", 1)[0]
		else:
			base = stem
		groups[base].append(item)

	# Build rename plan: within each base, patch index continuous
	rename_plan: List[Tuple[Path, str]] = []
	for base, group_items in groups.items():
		# sort by old name to keep order stable
		group_items.sort(key=lambda x: x.get("name", ""))
		for idx, item in enumerate(group_items, start=1):
			old_name = item.get("name")
			if not old_name:
				continue
			old_path = local_dir / old_name
			if not old_path.exists():
				continue
			suffix = old_path.suffix
			new_name = f"{base}_{idx:04d}{suffix}"
			rename_plan.append((old_path, new_name))
			item["name"] = new_name

	# First pass: rename to temp to avoid collisions
	tmp_plan: List[Tuple[Path, str]] = []
	for old_path, new_name in rename_plan:
		tmp_path = old_path.with_name(f".{old_path.stem}.tmp{old_path.suffix}")
		os.replace(old_path, tmp_path)
		tmp_plan.append((tmp_path, new_name))

	# Second pass: temp -> final
	for tmp_path, new_name in tmp_plan:
		os.replace(tmp_path, local_dir / new_name)


def load_ratio_map(path: Path | None) -> Dict:
	if not path:
		return {}
	if not path.exists():
		raise FileNotFoundError(f"Ratio file not found: {path}")
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		raise ValueError(f"Ratio file must be a JSON object: {path}")
	return data


def _normalize_dataset_ratio_map(data: Dict) -> Dict[str, float]:
	if not data:
		return {}
	if "datasets" in data and isinstance(data["datasets"], dict):
		data = data["datasets"]
	return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


def _get_field_ratio_map(dataset: str, data: Dict) -> Dict[str, float]:
	if not data:
		return {}
	# Supported formats:
	# 1) {"FIELD": {"DATASET": {label: ratio}}}
	# 2) {"FIELD": {"global": {label: ratio}}}
	# 3) {"DATASET": {label: ratio}}
	# 4) {label: ratio}
	if dataset in data and isinstance(data[dataset], dict):
		data = data[dataset]
	elif "global" in data and isinstance(data["global"], dict):
		data = data["global"]
	elif all(isinstance(v, (int, float)) for v in data.values()):
		# global flat mapping
		pass
	else:
		return {}
	return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


def _safe_label(value: object) -> str:
	if value is None:
		return "Unknown"
	if isinstance(value, str):
		v = value.strip()
		return v if v else "Unknown"
	return str(value)


def _item_keep_ratio(dataset: str, item: Dict, label_ratios: Dict, default_ratio: float) -> float:
	"""Get keep ratio for one item based on field configs (min across fields)."""
	if not label_ratios:
		return default_ratio

	keep_ratio = default_ratio
	for field in ("label", "material", "function", "traversability"):
		field_cfg = label_ratios.get(field)
		if not isinstance(field_cfg, dict):
			continue
		ratio_map = _get_field_ratio_map(dataset, field_cfg)
		if not ratio_map:
			continue
		value = _safe_label(item.get(field))
		if value in ratio_map:
			keep_ratio = min(keep_ratio, float(ratio_map[value]))
	return keep_ratio


def _weighted_sample(items: List[Dict], weights: List[float], k: int, rng: random.Random) -> List[Dict]:
	"""Sample k items without replacement using weights (Efraimidis-Spirakis)."""
	if k <= 0:
		return []
	if k >= len(items):
		return list(items)

	scored: List[Tuple[float, Dict]] = []
	for item, w in zip(items, weights):
		if w <= 0:
			continue
		u = rng.random()
		# Higher score => higher chance
		score = u ** (1.0 / w)
		scored.append((score, item))

	scored.sort(key=lambda x: x[0], reverse=True)
	return [item for _, item in scored[:k]]


def downsample_local_image_select(
	root: Path,
	datasets: Sequence[str] | None = None,
	dataset_ratios: Dict[str, float] | None = None,
	label_ratios: Dict | None = None,
	default_dataset_ratio: float = 1.0,
	default_label_weight: float = 1.0,
	global_total: int | None = None,
	seed: int = 42,
	dry_run: bool = False,
) -> Dict[str, Dict[str, int]]:
	local_dirs = find_local_image_select_dirs(root, datasets=datasets)
	if not local_dirs:
		print("[INFO] No local_image_select_4 directories found.")
		return {}

	rng = random.Random(seed)

	# First pass: collect dataset/split info
	entries = []
	dataset_totals: Dict[str, int] = defaultdict(int)
	for local_dir in local_dirs:
		dataset, split = dataset_and_split_from_dir(local_dir, root)
		label_path = local_dir / "local_label.json"
		label_data = load_local_label(label_path)
		items = label_data.get("items", [])
		if not isinstance(items, list):
			items = []
		# keep only items with existing image files
		valid_items: List[Dict] = []
		for item in items:
			name = item.get("name") if isinstance(item, dict) else None
			if not name:
				continue
			img_path = local_dir / name
			if img_path.exists() and img_path.suffix.lower() in IMAGE_EXTS:
				valid_items.append(item)

		entries.append({
			"local_dir": local_dir,
			"dataset": dataset,
			"split": split,
			"items": valid_items,
		})
		dataset_totals[dataset] += len(valid_items)

	# Compute dataset targets
	normalized_dataset_ratios = _normalize_dataset_ratio_map(dataset_ratios)
	dataset_weights: Dict[str, float] = {}
	for dataset in dataset_totals.keys():
		w = normalized_dataset_ratios.get(dataset, default_dataset_ratio)
		if w is None:
			w = default_dataset_ratio
		dataset_weights[dataset] = float(w)

	if global_total is not None:
		total_weight = sum(w for w in dataset_weights.values() if w > 0)
		dataset_targets = {
			ds: (0 if dataset_weights[ds] <= 0 else int(round(global_total * dataset_weights[ds] / total_weight)))
			for ds in dataset_weights
		}
	else:
		# default: keep ratio per dataset
		dataset_targets = {
			ds: int(round(dataset_totals[ds] * max(dataset_weights[ds], 0.0)))
			for ds in dataset_weights
		}

	stats: Dict[str, Dict[str, int]] = {}

	for entry in entries:
		local_dir = entry["local_dir"]
		dataset = entry["dataset"]
		split = entry["split"]
		items = entry["items"]
		key = f"{dataset}/{split}"
		total = len(items)
		if total == 0:
			stats[key] = {"total": 0, "kept": 0, "deleted": 0}
			continue

		# Split target proportional to this split's share in dataset
		ds_total = dataset_totals[dataset]
		if ds_total <= 0:
			target_total = 0
		else:
			ds_target = min(dataset_targets.get(dataset, 0), ds_total)
			target_total = int(round(ds_target * (total / ds_total)))
		target_total = min(target_total, total)

		# Compute per-item keep ratios
		item_ratios: List[float] = []
		for item in items:
			r = _item_keep_ratio(dataset, item, label_ratios, default_label_weight)
			item_ratios.append(max(0.0, min(1.0, r)))

		selected_items: List[Dict] = []
		if global_total is None:
			# Keep by ratio
			for item, r in zip(items, item_ratios):
				if r <= 0:
					continue
				if rng.random() <= r:
					selected_items.append(item)
			# If dataset ratio < 1, apply additionally by scaling
			ds_ratio = max(0.0, min(1.0, dataset_weights.get(dataset, 1.0)))
			if ds_ratio < 1.0 and selected_items:
				target = int(round(len(selected_items) * ds_ratio))
				selected_items = _weighted_sample(selected_items, [1.0] * len(selected_items), target, rng)
		else:
			# Weighted sample to target_total
			selected_items = _weighted_sample(items, item_ratios, target_total, rng)

		kept = len(selected_items)
		deleted = total - kept
		stats[key] = {"total": total, "kept": kept, "deleted": deleted}

		if dry_run:
			continue

		# Copy selected images to local_image_final
		output_dir = local_dir.parent / "local_image_final"
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

		selected_names = set()
		for item in selected_items:
			name = item.get("name")
			if not name:
				continue
			src = local_dir / name
			if not src.exists():
				continue
			try:
				shutil.copy2(src, output_dir / name)
				selected_names.add(name)
			except Exception as e:
				print(f"[WARN] Failed to copy {src}: {e}")

		# Save local_label.json for selected
		new_items = [item for item in selected_items if item.get("name") in selected_names]
		label_data = {"items": new_items}
		save_local_label(output_dir / "local_label.json", label_data)
		reindex_local_images(output_dir, label_data)
		save_local_label(output_dir / "local_label.json", label_data)

	return stats

