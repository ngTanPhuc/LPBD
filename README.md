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
```

If a new dataset or model is added, upload it to Kaggle and update `scripts/download_artifacts.sh` with the new Kaggle slug.

## Current Model Targets

This project is intended to evaluate:

- SAM 3
- YOLO26-seg
- FTW baseline model

## Status

This repository is in the early benchmark setup stage. The current focus is preparing reproducible data access, model artifact setup, notebook exploration, and baseline segmentation experiments.
