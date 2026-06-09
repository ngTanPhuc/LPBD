#!usr/bin/env bash

mkdir -p data/raw
mkdir -p models/pretrained

kaggle datasets download -d ngtanphuc020505/land-parcel-boundary-delineation -p data/raw --unzip
kaggle datasets download -d ngtanphuc020505/sam-3_pretrained -p models/pretrained --unzip