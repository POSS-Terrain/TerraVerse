#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image
from tqdm import tqdm



try:
    from ._dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root
except ImportError:
    from _dataset_paths import dataset_root as _shared_dataset_root, metadata_path as _shared_metadata_path, processed_root as _shared_processed_root

_DATASET_NAME = 'VAST'


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

TARGET_SIZE = (224, 224)


def ensure_clean_dir(path: Path):
	if path.exists():
		for p in path.iterdir():
			if p.is_file() or p.is_symlink():
				p.unlink(missing_ok=True)
			elif p.is_dir():
				shutil.rmtree(p, ignore_errors=True)
	else:
		path.mkdir(parents=True, exist_ok=True)


def process(global_dir: Path, local_dir: Path, limit: Optional[int] = None, reset: bool = True):
	list_path = global_dir / "global_list.json"
	if not list_path.exists():
		raise FileNotFoundError(f"Missing global_list.json at {list_path}")

	if reset:
		ensure_clean_dir(local_dir)
	else:
		local_dir.mkdir(parents=True, exist_ok=True)

	with list_path.open("r", encoding="utf-8") as f:
		payload = json.load(f)
	items = payload.get("items", [])

	saved = []
	processed = 0
	iter_items = items if limit is None else items[:limit]

	for entry in tqdm(iter_items, desc="processing", total=len(iter_items)):
		img_name = entry.get("img")
		label = entry.get("label")
		if not img_name or not label:
			continue

		img_path = global_dir / img_name
		if not img_path.exists():
			continue

		try:
			stem = Path(img_name).stem
			suffix = Path(img_name).suffix
			new_name = f"{stem}_0001{suffix}"
			img = Image.open(img_path).convert("RGB")
			img = img.resize(TARGET_SIZE, Image.BICUBIC)
			save_path = local_dir / new_name
			save_path.parent.mkdir(parents=True, exist_ok=True)
			img.save(save_path)
			saved.append({"name": new_name, "label": label})
			processed += 1
		except Exception:
			continue

	(local_dir / "local_label.json").write_text(
		json.dumps({"items": saved}, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	return len(saved)


def main(argv: Sequence[str] | None = None):
	parser = argparse.ArgumentParser(description="VAST global to local image processing")
	parser.add_argument(
		"--processed-root",
		type=Path,
		default=(_default_processed_root()),
		help="Path to processed_data root (contains global_image/local_image)",
	)
	parser.add_argument(
		"--limit-image",
		type=int,
		default=None,
		help="Optional limit of images to process (for quick test)",
	)
	parser.add_argument("--no-reset", action="store_true", help="Do not clear local_image folder")
	args = parser.parse_args(list(argv) if argv is not None else None)

	global_dir = args.processed_root / "global_image"
	local_dir = args.processed_root / "local_image"

	count = process(global_dir, local_dir, limit=args.limit_image, reset=(not args.no_reset))
	print(f"Saved {count} resized images -> {local_dir}")


def run(extra_args: Sequence[str] | None = None) -> None:
    main(extra_args)


if __name__ == "__main__":
    main()
