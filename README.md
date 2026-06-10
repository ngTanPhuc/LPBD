# LPBD

LPBD stands for Land Parcel Boundary Delineation. This repository is an experimental benchmark for preparing data, running segmentation models, and comparing land parcel boundary extraction results.

The large dataset files and model checkpoints are not stored in GitHub. They are downloaded from Kaggle during project setup.

## Project Structure

```text
LPBD/
├── configs/        # Experiment and model configuration files
├── data/           # Local datasets; ignored by Git except .gitkeep
├── experiments/    # Experiment outputs and run records
├── models/         # Local model checkpoints; ignored by Git except .gitkeep
├── notebooks/      # Exploration, data checks, and model test notebooks
├── reports/        # Evaluation reports and result summaries
├── scripts/        # Utility scripts for setup and data processing
├── src/            # Source code
├── tests/          # Tests
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11 or newer is recommended.
- A Kaggle account is required to download the dataset and model artifacts.
- `pip` must be available in your Python environment.

## Setup On A New Machine

Clone the repository:

```bash
git clone git@github.com:ngTanPhuc/LPBD.git
cd LPBD
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Kaggle Authentication

The project uses the Kaggle CLI to download artifacts. Create a Kaggle API token first:

1. Go to your Kaggle account settings.
2. Click `Create New API Token`.
3. Kaggle downloads a file named `kaggle.json`.

Move the token into the location expected by the Kaggle CLI:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

If `kaggle.json` is somewhere else, adjust the source path in the `mv` command.

## Download Dataset And Models

Run the artifact download script:

```bash
bash scripts/download_artifacts.sh
```

The script downloads:

```bash
kaggle datasets download -d ngtanphuc020505/land-parcel-boundary-delineation -p data/raw --unzip
kaggle datasets download -d ngtanphuc020505/sam-3_pretrained -p models/pretrained --unzip
```

After this step, the expected local folders are:

```text
data/raw/              # Dataset files from Kaggle
models/pretrained/     # Pretrained model files from Kaggle
```

These folders are intentionally ignored by Git because they contain large artifacts.

## Notebook Usage

Start Jupyter from the activated environment:

```bash
python -m ipykernel install --user --name lpbd --display-name "Python (LPBD)"
jupyter notebook
```

Then open notebooks from the `notebooks/` directory and select the `Python (LPBD)` kernel.

## Artifact Policy

Do not commit datasets, checkpoints, or generated experiment artifacts to Git. The `.gitignore` keeps these paths local:

```gitignore
data/*
!data/.gitkeep

models/*
!models/.gitkeep

external/*
!external/.gitkeep
```

If a new dataset or model is added, upload it to Kaggle and update `scripts/download_artifacts.sh` with the new Kaggle slug.

## Current Model Targets

This project is intended to evaluate:

- SAM 3
- YOLO26-seg
- FTW baseline model

## SAM 3

The SAM 3 fine-tuning config is stored in:

```text
configs/sam3/lpbd_ftw.yaml
```

The config is written for the filtered FTW Vietnam dataset prepared by `scripts/prepare_sam3_ftw_dataset.py`. The prepared dataset should look like this:

```text
data/processed/sam3_ftw/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── images/
    ├── train/
    ├── val/
    └── test/
```

The current config assumes the filtered `<=200` parcels per image dataset:

```text
train: 45 images, 7556 annotations
val:   12 images, 1687 annotations
test:  3 images, 476 annotations
```

### Train SAM 3 On Kaggle

The GitHub repo does not include `data/`, `models/`, or `external/sam3/`. On Kaggle, you must recreate or attach these artifacts before training.

Recommended Kaggle setup:

1. Create a Kaggle Notebook with GPU enabled.
2. Attach the LPBD raw dataset Kaggle Dataset.
3. Attach the SAM 3 pretrained checkpoint Kaggle Dataset.
4. Clone this repo into `/kaggle/working`.
5. Clone the SAM 3 repo into `external/sam3`.
6. Prepare the filtered SAM 3 dataset.
7. Copy the LPBD config into SAM 3's Hydra config folder.
8. Launch training.

Example Kaggle commands:

```bash
cd /kaggle/working
git clone https://github.com/ngTanPhuc/LPBD.git
cd LPBD

pip install --upgrade pip
pip install -r requirements.txt

git clone https://github.com/facebookresearch/sam3.git external/sam3
pip install -e "external/sam3[train]"
```

Kaggle mounts attached datasets under `/kaggle/input`. Check the exact folder names first:

```bash
find /kaggle/input -maxdepth 3 -type d
```

Copy or arrange the raw FTW Vietnam dataset so this path exists:

```text
data/raw/FTW_Vietnam/
```

For example, if Kaggle mounts the raw dataset at `/kaggle/input/land-parcel-boundary-delineation/FTW_Vietnam`, run:

```bash
mkdir -p data/raw
cp -r /kaggle/input/land-parcel-boundary-delineation/FTW_Vietnam data/raw/FTW_Vietnam
```

Prepare the filtered SAM 3 dataset:

```bash
python scripts/prepare_sam3_ftw_dataset.py --clean-output --overwrite
```

This creates:

```text
data/processed/sam3_ftw/
```

Copy the config into SAM 3's config folder:

```bash
cp configs/sam3/lpbd_ftw.yaml external/sam3/sam3/train/configs/lpbd_ftw.yaml
```

The config expects the SAM 3 checkpoint at:

```text
/kaggle/input/sam3-checkpoint/sam3.pt
```

Check the actual checkpoint path in your Kaggle notebook:

```bash
find /kaggle/input -name "sam3.pt"
```

If the returned path is different, update `paths.checkpoint_path` in `configs/sam3/lpbd_ftw.yaml`, then copy the config again:

```bash
cp configs/sam3/lpbd_ftw.yaml external/sam3/sam3/train/configs/lpbd_ftw.yaml
```

Train from the SAM 3 repo:

```bash
cd external/sam3

python sam3/train/train.py \
  -c configs/lpbd_ftw.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

Training outputs will be written to:

```text
/kaggle/working/LPBD/experiments/sam3_ftw_le200_full_finetune/
```

### SAM 3 Config Notes

The current config uses:

```yaml
scratch:
  num_queries: 200
  max_ann_per_img: 200
  train_batch_size: 1
  val_batch_size: 1
  target_epoch_size: 45
  max_data_epochs: 50
```

This is intentional because the dataset preparation script filters out chips with more than 200 parcel instances. If you train on unfiltered chips, update both the dataset and config; otherwise SAM 3 may not have enough queries for all parcels in dense images.

## Status

This repository is in the early benchmark setup stage. The current focus is preparing reproducible data access, model artifact setup, notebook exploration, and baseline segmentation experiments.
