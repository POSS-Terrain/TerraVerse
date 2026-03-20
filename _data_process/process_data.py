#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence


EXCLUDED_DATASETS = {"Jackal", "TerraPOSS", "RSCD"}
DEFAULT_DATASETS = [
    "ACDC",
    "COCO-Stuff",
    "DeepScene",
    "FCDD",
    "GOOSE",
    "GOOSE-Ex",
    "IDD",
    "KITTI-360",
    "ORAD-3D",
    "ORAD-3D-Label",
    "ORFD",
    "RELLIS",
    "RTK",
    "RUGD",
    "TAS500",
    "VAST",
    "WildScenes",
    "YCOR",
]

DEFAULT_DBCNN_THRESHOLDS = {
    "ACDC": 0.3,
    "COCO-Stuff": 0.5,
    "DeepScene": 0.225,
    "FCDD": 0.3,
    "GOOSE": 0.25,
    "GOOSE-Ex": 0.3,
    "GOOSE/goose": 0.25,
    "GOOSE/gooseEx": 0.3,
    "IDD": 0.3,
    "KITTI-360": 0.225,
    "ORAD-3D": 0.3,
    "RTK": 0.275,
    "RELLIS": 0.3,
    "RUGD": 0.225,
    "TAS500": 0.25,
    "VAST": 0.225,
    "WildScenes": 0.275,
    "YCOR": 0.3,
}

DEFAULT_PIQE_THRESHOLDS = {
    "FCDD": 60.0,
    "KITTI-360": 60.0,
    "RTK": 50.0,
    "RUGD": 60.0,
    "VAST": 60.0,
    "WildScenes": 60.0,
}

STEP_ORDER = [
    "mask_to_patch",
    "label_clean",
    "dbcnn",
    "piqe",
    "other",
    "deduplication",
    "downsample",
]

INTERMEDIATE_DIRS = [
    "local_image_select",
    "local_image_select_2",
    "local_image_select_3",
    "local_image_select_4",
]

AUXILIARY_FILES = [
    "quality_kept.csv",
    "quality_deleted.csv",
]

AUXILIARY_DIRS = [
    "Z_image_score",
    "L_image_score",
]


def normalize_datasets(datasets: Optional[Sequence[str]]) -> list[str]:
    if not datasets:
        return list(DEFAULT_DATASETS)
    return [dataset for dataset in datasets if dataset not in EXCLUDED_DATASETS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TerraVerse data processing pipeline and keep only the final local_image outputs"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
        help="Dataset root directory. Defaults to TerraVerse_Released/data.",
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        default=None,
        help="Process only the specified dataset names relative to --root (multiple allowed).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions without writing files")
    parser.add_argument(
        "--steps",
        nargs="*",
        choices=STEP_ORDER,
        default=STEP_ORDER,
        help="Steps to run. Defaults to the full pipeline.",
    )
    parser.add_argument("--limit-image", type=int, default=None, help="Maximum number of images to keep per dataset in the label_clean step")
    parser.add_argument("--label-select", type=Path, default=Path(__file__).resolve().parent / "label_select.json")
    parser.add_argument("--dataset-ratios", type=Path, default=None, help="JSON file with per-dataset downsample ratios")
    parser.add_argument("--total", type=int, default=None, help="Global target count for downsampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for downsampling")
    parser.add_argument("--clean-output", action="store_true", help="Clean previous intermediate directories and outputs before running")
    return parser.parse_args()


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def summarize_stats(step: str, stats: Dict[str, Dict[str, int]]) -> None:
    if not stats:
        print(f"[{step}] No results to summarize")
        return
    print(f"[{step}] Summary")
    for key, value in stats.items():
        parts = ", ".join(f"{k}={v}" for k, v in value.items())
        print(f"  - {key}: {parts}")


def parse_stage_dir(root: Path, dir_path: Path) -> tuple[str, str, Path] | None:
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


def iter_stage_dirs(root: Path, folder_name: str, datasets: Optional[Sequence[str]]):
    dataset_filter = set(datasets) if datasets else None
    for stage_dir in root.rglob(folder_name):
        if not stage_dir.is_dir():
            continue
        info = parse_stage_dir(root, stage_dir)
        if info is None:
            continue
        dataset, split, base_dir = info
        if dataset_filter and dataset not in dataset_filter:
            continue
        yield dataset, split, base_dir, stage_dir


def iter_base_dirs(root: Path, datasets: Optional[Sequence[str]]):
    seen = set()
    folder_names = list(INTERMEDIATE_DIRS) + ["local_image", "local_image_final"]
    for folder_name in folder_names:
        for _, _, base_dir, _ in iter_stage_dirs(root, folder_name, datasets):
            key = str(base_dir)
            if key in seen:
                continue
            seen.add(key)
            yield base_dir


def prune_empty_processed_data_dirs(root: Path, datasets: Optional[Sequence[str]]) -> None:
    dataset_filter = set(datasets) if datasets else None
    candidates = sorted(root.rglob("processed_data"), key=lambda p: len(p.parts), reverse=True)
    for processed_dir in candidates:
        info = None
        for dataset_dir in processed_dir.parents:
            if dataset_dir.parent == root:
                info = dataset_dir.name
                break
        if dataset_filter and info and info not in dataset_filter:
            continue
        if processed_dir.is_dir() and not any(processed_dir.iterdir()):
            remove_dir(processed_dir)


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"[CLEAN] removed {path}")


def remove_file(path: Path) -> None:
    if path.exists():
        path.unlink()
        print(f"[CLEAN] removed {path}")


def cleanup_generated_content(root: Path, datasets: Optional[Sequence[str]], remove_final: bool) -> None:
    folders_to_remove = list(INTERMEDIATE_DIRS) + ["local_image_final"]
    if remove_final:
        folders_to_remove.append("local_image")

    for folder_name in folders_to_remove:
        for _, _, _, stage_dir in list(iter_stage_dirs(root, folder_name, datasets)):
            remove_dir(stage_dir)

    dataset_filter = set(datasets) if datasets else None
    for file_name in AUXILIARY_FILES:
        for file_path in root.rglob(file_name):
            parent_info = parse_stage_dir(root, file_path.parent)
            if parent_info is None:
                continue
            dataset, _, _ = parent_info
            if dataset_filter and dataset not in dataset_filter:
                continue
            remove_file(file_path)

    for aux_dir_name in AUXILIARY_DIRS:
        aux_dir = root / aux_dir_name
        if aux_dir.is_dir():
            remove_dir(aux_dir)

    prune_empty_processed_data_dirs(root, datasets)


def cleanup_auxiliary_files(root: Path, datasets: Optional[Sequence[str]]) -> None:
    cleanup_generated_content(root, datasets, remove_final=False)


def finalize_outputs(root: Path, datasets: Optional[Sequence[str]]) -> int:
    moved_count = 0
    seen_targets = set()
    for _, _, base_dir, final_dir in list(iter_stage_dirs(root, "local_image_final", datasets)):
        target_dir = base_dir / "local_image"
        target_key = str(target_dir)
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        if target_dir.exists():
            shutil.rmtree(target_dir)
            print(f"[FINALIZE] replaced existing {target_dir}")
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(final_dir), str(target_dir))
        print(f"[FINALIZE] {final_dir} -> {target_dir}")
        moved_count += 1

    for _, _, _, final_dir in list(iter_stage_dirs(root, "local_image_final", datasets)):
        if final_dir.exists():
            remove_dir(final_dir)

    prune_empty_processed_data_dirs(root, datasets)
    return moved_count


def find_source_label_files(root: Path, datasets: Optional[Sequence[str]]) -> list[Path]:
    from label_clean import dataset_key_from_file

    label_files = []
    for label_file in sorted(root.rglob("local_label.json")):
        if label_file.parent.name not in {"local_image", "local_image_final"}:
            continue
        parent_info = parse_stage_dir(root, label_file.parent)
        if parent_info is None:
            continue
        ds_key = dataset_key_from_file(root, label_file)
        if datasets and ds_key not in set(datasets):
            continue
        label_files.append(label_file)

    unique_files = []
    seen = set()
    for path in label_files:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_files.append(path)
    return unique_files


def run_mask_to_patch_step(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    from mask_to_patch import HANDLERS

    stats: Dict[str, Dict[str, int]] = {}
    for dataset in args.dataset or []:
        if dataset not in HANDLERS:
            continue
        if args.dry_run:
            print(f"[DRY-RUN] mask_to_patch -> {dataset}")
            stats[dataset] = {"planned": 1}
            continue
        try:
            HANDLERS[dataset]([])
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"mask_to_patch 阶段无法处理数据集 {dataset}。\n{exc}"
            ) from exc
        stats[dataset] = {"executed": 1}
    return stats


def run_label_clean_step(root: Path, datasets: Optional[Sequence[str]], label_select_path: Path, limit_image: Optional[int], dry_run: bool) -> Dict[str, Dict[str, int]]:
    from label_clean import JsonLabelSink, dataset_key_from_file, load_label_select, process_dataset

    label_select = load_label_select(label_select_path)
    label_files = find_source_label_files(root, datasets)

    stats: Dict[str, Dict[str, int]] = {}
    if not label_files:
        print("[label_clean] No local_label.json found under local_image or local_image_final")
        return stats

    for label_file in label_files:
        ds_key = dataset_key_from_file(root, label_file)
        label_map = label_select.get(ds_key, {}).get("labels", {})
        if not label_map:
            print(f"[label_clean] Skip {ds_key}: no mapping found in label_select.json")
            continue

        output_dir = label_file.parent.parent / "local_image_select"
        if output_dir.exists() and not dry_run:
            shutil.rmtree(output_dir)
        label_sink = None if dry_run else JsonLabelSink(output_dir / "local_label.json")
        kept = process_dataset(
            label_file=label_file,
            output_dir=output_dir,
            label_map=label_map,
            limit_images=limit_image,
            label_sink=label_sink,
            dry_run=dry_run,
        )
        stats[label_file.relative_to(root).parent.as_posix()] = {"kept": kept}
        print(f"[label_clean] {label_file}: kept={kept}")

    return stats


def run_dbcnn_step(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    from image_clean_DBCNN import filter_local_image_select as run_dbcnn

    return run_dbcnn(
        root=args.root,
        datasets=args.dataset,
        model_name="dbcnn",
        quality_threshold=0.33,
        thresholds=DEFAULT_DBCNN_THRESHOLDS,
        do_delete=not args.dry_run,
        reindex=True,
        save_csv=False,
        save_histogram=False,
        hist_bins=50,
    )


def run_piqe_step(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    from image_clean_PIQE import filter_local_image_select as run_piqe

    return run_piqe(
        root=args.root,
        datasets=args.dataset,
        model_name="piqe",
        quality_threshold=40.0,
        thresholds=DEFAULT_PIQE_THRESHOLDS,
        do_delete=not args.dry_run,
        reindex=True,
        save_csv=False,
        save_histogram=False,
        hist_bins=50,
    )


def run_other_step(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    from image_clean_other import filter_local_image_select as run_other

    return run_other(
        root=args.root,
        datasets=args.dataset,
        quality_threshold=0.4,
        do_delete=not args.dry_run,
        reindex=True,
        save_csv=False,
        save_histogram=False,
        hist_bins=50,
        input_dir_name="local_image_select_2",
        output_dir_name="local_image_select_3",
    )


def run_dedup_step(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    from image_deduplication import deduplicate_local_images

    return deduplicate_local_images(
        root=args.root,
        datasets=args.dataset,
        input_dir_name="local_image_select_3",
        output_dir_name="local_image_select_4",
        chunk_size=10,
        method="dhash",
        threshold=5,
        dry_run=args.dry_run,
    )


def run_downsample_step(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    from image_downsample import downsample_local_image_select, load_ratio_map

    dataset_ratios = load_ratio_map(args.dataset_ratios) if args.dataset_ratios else {}
    return downsample_local_image_select(
        root=args.root,
        datasets=args.dataset,
        dataset_ratios=dataset_ratios,
        label_ratios=None,
        default_dataset_ratio=1.0,
        default_label_weight=1.0,
        global_total=args.total,
        seed=args.seed,
        dry_run=args.dry_run,
    )


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.dataset = normalize_datasets(args.dataset)

    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.root}")

    if args.clean_output and not args.dry_run:
        print_header("Clean previous intermediate directories and outputs")
        cleanup_generated_content(args.root, args.dataset, remove_final=True)

    step_handlers: Dict[str, Callable[[], Dict[str, Dict[str, int]]]] = {
        "mask_to_patch": lambda: run_mask_to_patch_step(args),
        "label_clean": lambda: run_label_clean_step(args.root, args.dataset, args.label_select, args.limit_image, args.dry_run),
        "dbcnn": lambda: run_dbcnn_step(args),
        "piqe": lambda: run_piqe_step(args),
        "other": lambda: run_other_step(args),
        "deduplication": lambda: run_dedup_step(args),
        "downsample": lambda: run_downsample_step(args),
    }

    print(f"root={args.root}")
    print(f"steps={args.steps}")
    print(f"dry_run={args.dry_run}")
    print(f"datasets={list(args.dataset)}")

    overall: Dict[str, Dict[str, Dict[str, int]]] = {}
    for step in STEP_ORDER:
        if step not in args.steps:
            continue
        print_header(f"STEP: {step}")
        stats = step_handlers[step]()
        overall[step] = stats
        summarize_stats(step, stats)

    if not args.dry_run and "downsample" in args.steps:
        print_header("Finalize output")
        moved = finalize_outputs(args.root, args.dataset)
        cleanup_auxiliary_files(args.root, args.dataset)
        print(f"[FINALIZE] local_image directories created: {moved}")
        print("[FINALIZE] Intermediate directories cleaned; only final local_image outputs remain")

    print_header("Pipeline finished")
    for step in args.steps:
        print(f"{step}: {len(overall.get(step, {}))} group(s)")


if __name__ == "__main__":
    main()
