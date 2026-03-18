# TerraVerse Released — Code Overview

This folder contains the code used to generate and process the released **TerraVerse** dataset, as well as the baseline scripts for **TerraBench-Pro**.

The content in this folder can be grouped into two main parts:

1. **Data processing pipeline** (generates `local_image/` patches from `global_image/` imagery)
2. **TerraBench-Pro baselines** (benchmark training/evaluation code for multiple model families)

---

## 🛠️ Code (Processing Pipeline)

All data processing code is located under `code/_data_process/`.

### Main entry point
To generate `local_image/` patches from `global_image/`, run:

```bash
python code/_data_process/process_data.py --dataset <datasetName>
```

This executes the full pipeline from `global_image/` → `local_image/` (currently the final annotation step is not automatically generated).

### Processing pipeline stages
The pipeline is structured as a sequence of stages, each implemented as an individual script under `code/_data_process/`:

1. `m2p` (metadata-to-patch / initial patch extraction)
2. `label-clean` (clean / normalize labels)
3. `DBCNN` (image quality / consistency filtering)
4. `PIQE` (image quality scoring)
5. `other` (additional heuristic filtering)
6. `deduplication` (remove duplicate patches)
7. `downsample` (optional downsampling / balancing)
8. `annotation` (annotation via MLLM)

### Where to find supporting code
- `code/_data_process/`: driver script + per-stage processing scripts
- `code/_data_process/utils/`: dataset-specific patch extraction helpers (e.g., `mask_to_patch_<dataset>.py`)

---

## 🧪 TerraBench-Pro Baselines (Benchmarks + Models)

This repository also includes the baseline benchmark code used in TerraBench-Pro.

### What’s included
- **Benchmark datasets organization** (folder structure and data loading logic)
- **Baseline model code** for multiple backbones (no pre-trained weights included)

### Model families included
- **MobileNet** (Chen et al.)
- **ResNet-18** (Hanson et al.)
- **EfficientNet** (Zhao et al.)
- **DINOv2** (self-supervised transformer)
- **Vanilla CLIP**
- **TerraCLIP** (our CLIP-style model)

Each of these has dedicated scripts for training and evaluation located in subdirectories such as:
- `1.Chen_et_al.[4]/`
- `2.Hanson_et_al.[5]/`
- `3.Zhao_et_al.[2]/`
- `4.DINOv2[15]/`
- `5.VanillaCLIP[14]/`
- `6.TerraCLIP[9]/`

---

## 🧩 Additional Notes
- Requirements: see `requirements.txt` for the full list of Python dependencies.
- Annotation helpers: `code/_annotation/annotation.py` contains code used for generating/formatting annotations.

---


