#!/usr/bin/env bash

set -euo pipefail

kaggle datasets download -d ngtanphuc020505/land-parcel-boundary-delineation -p data/raw --unzip
kaggle models instances versions download ngtanphuc020505/sam-3-pretrained/pytorch/default/1 -p models/pretrained/sam3

tar -xzf models/pretrained/sam3/*.tar.gz -C models/pretrained/sam3
rm models/pretrained/sam3/*.tar.gz