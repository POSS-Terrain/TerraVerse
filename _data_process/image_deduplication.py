from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def find_local_image_dirs(root: Path, datasets: Sequence[str] | None, folder: str) -> List[Path]:
        dirs: List[Path] = []
        dataset_filter = set(datasets) if datasets else None
        for p in root.rglob(folder):
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


def list_images(folder: Path) -> List[Path]:
	return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def base_key_from_name(name: str) -> str:
	stem = Path(name).stem
	if "_" in stem:
		return stem.rsplit("_", 1)[0]
	return stem


def ahash(img: Image.Image, size: int = 8) -> np.ndarray:
	img = img.convert("L").resize((size, size), Image.LANCZOS)
	arr = np.asarray(img, dtype=np.float32)
	return arr > arr.mean()


def dhash(img: Image.Image, size: int = 8) -> np.ndarray:
	img = img.convert("L").resize((size + 1, size), Image.LANCZOS)
	arr = np.asarray(img, dtype=np.float32)
	return arr[:, 1:] > arr[:, :-1]


def hist_feature(img: Image.Image, bins: int = 256) -> np.ndarray:
	img = img.convert("L")
	arr = np.asarray(img, dtype=np.uint8)
	hist = np.bincount(arr.flatten(), minlength=bins).astype(np.float32)
	if hist.sum() > 0:
		hist /= hist.sum()
	return hist


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
	return int(np.count_nonzero(h1 != h2))


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
	den = (np.linalg.norm(v1) * np.linalg.norm(v2))
	if den == 0:
		return 0.0
	return float(np.dot(v1, v2) / den)


def is_duplicate(img_a: Image.Image, img_b: Image.Image, method: str, threshold: float) -> bool:
	if method == "ahash":
		return hamming_distance(ahash(img_a), ahash(img_b)) <= int(threshold)
	if method == "dhash":
		return hamming_distance(dhash(img_a), dhash(img_b)) <= int(threshold)
	if method == "hist":
		return cosine_similarity(hist_feature(img_a), hist_feature(img_b)) >= threshold
	raise ValueError(f"Unsupported method: {method}")


def deduplicate_chunk(
	image_paths: List[Path],
	method: str,
	threshold: float,
) -> Tuple[List[Path], List[Path]]:
	"""Keep first, remove duplicates within chunk."""
	kept: List[Path] = []
	removed: List[Path] = []

	for path in image_paths:
		try:
			img = Image.open(path)
			img.load()
		except Exception:
			removed.append(path)
			continue

		is_dup = False
		for kept_path in kept:
			try:
				kimg = Image.open(kept_path)
				kimg.load()
			except Exception:
				continue
			if is_duplicate(img, kimg, method, threshold):
				is_dup = True
				break

		if is_dup:
			removed.append(path)
		else:
			kept.append(path)

	return kept, removed


def deduplicate_local_images(
	root: Path,
	datasets: Sequence[str] | None,
	input_dir_name: str,
	output_dir_name: str,
	chunk_size: int,
	method: str,
	threshold: float,
	dry_run: bool,
) -> Dict[str, Dict[str, int]]:
	local_dirs = find_local_image_dirs(root, datasets, input_dir_name)
	if not local_dirs:
		print(f"[INFO] No {input_dir_name} folders found.")
		return {}

	stats: Dict[str, Dict[str, int]] = {}
	for local_dir in local_dirs:
		dataset, split = dataset_and_split_from_dir(local_dir, root)
		key = f"{dataset}/{split}"
		image_paths = list_images(local_dir)

		removed_names = set()
		kept_count = 0
		removed_count = 0
		paths = sorted(image_paths)
		for i in tqdm(range(0, len(paths), chunk_size), desc=key):
			chunk = paths[i:i + chunk_size]
			kept, removed = deduplicate_chunk(chunk, method, threshold)
			kept_count += len(kept)
			removed_count += len(removed)
			for rp in removed:
				removed_names.add(rp.name)

		if not dry_run:
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

			for p in image_paths:
				if p.name in removed_names:
					continue
				try:
					from shutil import copy2

					copy2(p, output_dir / p.name)
				except Exception as e:
					print(f"[WARN] Failed to copy {p}: {e}")

			label_path = local_dir / "local_label.json"
			label_data = load_local_label(label_path)
			items = label_data.get("items", [])
			if isinstance(items, list):
				new_items = [item for item in items if item.get("name") not in removed_names]
				label_data["items"] = new_items
				save_local_label(output_dir / "local_label.json", label_data)
				reindex_local_images(output_dir, label_data)
				save_local_label(output_dir / "local_label.json", label_data)
		else:
			if removed_names:
				print(f"[{key}] removed ids (dry-run):")
				for name in sorted(removed_names):
					print(f"  {name}")

		stats[key] = {
			"total": len(image_paths),
			"kept": kept_count,
			"removed": removed_count,
		}

	return stats

