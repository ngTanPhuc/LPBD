#!usr/bin/env bash

mkdir -p data/raw
mkdir -p models/pretrained

kaggle datasets download -d ngtanphuc020505/land-parcel-boundary-delineation -p data/raw --unzip
kaggle models instances download ngtanphuc020505/sam-3-pretrained/pytorch/sam3-h/1 -p models/pretrained --unzip