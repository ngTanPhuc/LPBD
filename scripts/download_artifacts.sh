#!/usr/bin/env bash

set -euo pipefail

kaggle datasets download -d ngtanphuc020505/land-parcel-boundary-delineation -p data/raw --unzip
kaggle models get ngtanphuc020505/sam-3-pretrained -p models/pretrained/sam3
