#!/usr/bin/env bash

set -euo pipefail

mkdir -p models/sam3

kaggle datasets download -d ngtanphuc020505/land-parcel-boundary-delineation -p data/raw --unzip
kaggle models instances versions download ngtanphuc020505/sam-3-pretrained/pytorch/default/1 -p models/pretrained/sam3
