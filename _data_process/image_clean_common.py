from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_local_label(path: Path) -> Dict:
    if not path.exists():
        return {"items": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def save_local_label(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def reindex_local_images(local_dir: Path, label_data: Dict) -> None:
    items = label_data.get("items", [])
    if not isinstance(items, list) or not items:
        return

    groups: Dict[str, List[Dict]] = defaultdict(list)
    for item in items:
        name = item.get("name")
        if not name:
            continue
        stem = Path(name).stem
        base = stem.rsplit("_", 1)[0] if "_" in stem else stem
        groups[base].append(item)

    rename_plan: List[Tuple[Path, str]] = []
    for base, group_items in groups.items():
        group_items.sort(key=lambda value: value.get("name", ""))
        for idx, item in enumerate(group_items, start=1):
            old_name = item.get("name")
            if not old_name:
                continue
            old_path = local_dir / old_name
            if not old_path.exists():
                continue
            new_name = f"{base}_{idx:04d}{old_path.suffix}"
            rename_plan.append((old_path, new_name))
            item["name"] = new_name

    tmp_plan: List[Tuple[Path, str]] = []
    for old_path, new_name in rename_plan:
        tmp_path = old_path.with_name(f".{old_path.stem}.tmp{old_path.suffix}")
        os.replace(old_path, tmp_path)
        tmp_plan.append((tmp_path, new_name))

    for tmp_path, new_name in tmp_plan:
        os.replace(tmp_path, local_dir / new_name)



def save_records_to_csv(path: Path, records: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "score", "full_path", "split", "dataset"])
        return

    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)



def save_score_histogram(scores: List[float], output_path: Path, bins: int = 50) -> None:
    if not scores:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skip histogram generation.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=bins)
    plt.xlabel("score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
