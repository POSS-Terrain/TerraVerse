import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class LabelEntry:
	name: str
	label: str
	meta: Dict


class LabelSink:
	"""Placeholder interface for saving filtered labels/metadata."""

	def add(self, entry: LabelEntry, new_name: str, dst_path: Path, label_info: Dict) -> None:
		pass

	def finalize(self) -> None:
		pass


class JsonLabelSink(LabelSink):
	def __init__(self, output_json: Path) -> None:
		self.output_json = output_json
		self.items: List[Dict] = []

	def add(self, entry: LabelEntry, new_name: str, dst_path: Path, label_info: Dict) -> None:
		self.items.append(
			{
				"name": new_name,
				"label": entry.label,
				"top_left": entry.meta.get("top_left"),
				"bottom_right": entry.meta.get("bottom_right"),
				"category": label_info.get("category"),
				"material": label_info.get("material"),
				"function": label_info.get("function"),
				"traversability": label_info.get("traversability"),
			}
		)

	def finalize(self) -> None:
		self.output_json.parent.mkdir(parents=True, exist_ok=True)
		payload = {"items": self.items}
		with self.output_json.open("w", encoding="utf-8") as f:
			json.dump(payload, f, ensure_ascii=False, indent=2)


def find_label_files(root: Path) -> List[Path]:
        """Find all local_label.json files under any local_image path."""
        results: List[Path] = []
        for path in root.rglob("local_label.json"):
                if path.parent.name != "local_image":
                        continue
                try:
                        path.relative_to(root)
                except ValueError:
                        continue
                results.append(path)
        return sorted(results)


def dataset_key_from_file(root: Path, file_path: Path) -> str:
        """Derive dataset key from either direct split layout or legacy processed_data layout."""
        rel = file_path.relative_to(root)
        parts = list(rel.parts)
        try:
                idx = parts.index("processed_data")
        except ValueError:
                return parts[0] if parts else str(rel)
        return "/".join(parts[:idx])


def load_label_json(file_path: Path) -> object:
	with file_path.open("r", encoding="utf-8") as f:
		return json.load(f)


def extract_label_entries(data: object) -> Iterable[LabelEntry]:
	"""
	Extract LabelEntry records from parsed JSON.
	TODO: Customize this function to match actual JSON schema.
	"""
	if isinstance(data, list):
		for item in data:
			if isinstance(item, dict) and "name" in item and "label" in item:
				yield LabelEntry(name=item["name"], label=item["label"], meta=item)
		return
	if isinstance(data, dict):
		items = data.get("items")
		if isinstance(items, list):
			for item in items:
				if isinstance(item, dict) and "name" in item and "label" in item:
					yield LabelEntry(name=item["name"], label=item["label"], meta=item)


def load_label_select(file_path: Path) -> Dict[str, Dict[str, Dict]]:
	if not file_path.exists():
		return {}
	with file_path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		return {}
	return data


def process_image(src_path: Path) -> Path:
	"""
	Placeholder for image processing.
	Return the path to the file to be saved (default: original file).
	"""
	return src_path


def base_image_key(name: str) -> str:
	"""Return a key for the original image to group patches."""
	stem = Path(name).stem
	if "_" in stem:
		return stem.rsplit("_", 1)[0]
	return stem


def next_name(base_key: str, patch_index: int) -> str:
	return f"{base_key}_{patch_index:04d}"


def ensure_suffix(src_path: Path, new_stem: str) -> str:
	return f"{new_stem}{src_path.suffix}"


def copy_processed_image(src_path: Path, dst_path: Path) -> None:
	shutil.copy2(src_path, dst_path)


def process_dataset(
	label_file: Path,
	output_dir: Path,
	label_map: Dict[str, Dict],
	limit_images: Optional[int] = None,
	label_sink: Optional[LabelSink] = None,
	dry_run: bool = False,
) -> int:
	local_image_dir = label_file.parent
	if not dry_run:
		output_dir.mkdir(parents=True, exist_ok=True)

	data = load_label_json(label_file)
	entries = list(extract_label_entries(data))
	total_entries = len(entries)

	base_to_patch: Dict[str, int] = {}
	kept_count = 0

	for idx, entry in enumerate(entries, start=1):
		if not entry.label or not entry.name:
			continue
		label_key = str(entry.label).strip()
		label_info = label_map.get(label_key)
		if not label_info:
			continue

		src_path = local_image_dir / entry.name
		if not src_path.exists():
			continue

		base_key = base_image_key(entry.name)
		if base_key not in base_to_patch:
			base_to_patch[base_key] = 0

		base_to_patch[base_key] += 1
		new_stem = next_name(base_key, base_to_patch[base_key])
		new_name = ensure_suffix(src_path, new_stem)
		dst_path = output_dir / new_name

		processed_path = process_image(src_path)
		if not dry_run:
			copy_processed_image(processed_path, dst_path)

		if label_sink is not None and not dry_run:
			clean_entry = LabelEntry(name=entry.name, label=label_key, meta=entry.meta)
			label_sink.add(clean_entry, new_name, dst_path, label_info)
		kept_count += 1
		if limit_images is not None and kept_count >= limit_images:
			break

		if total_entries > 0 and (idx == total_entries or idx % 200 == 0):
			msg = f"[PROGRESS] {label_file.name}: {idx}/{total_entries} entries, kept {kept_count}"
			sys.stdout.write("\r" + msg + " " * 10)
			sys.stdout.flush()

	if label_sink is not None and not dry_run:
		label_sink.finalize()
	if total_entries > 0:
		sys.stdout.write("\n")
		sys.stdout.flush()

	return kept_count

