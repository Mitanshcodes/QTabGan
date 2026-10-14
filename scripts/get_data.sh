#!/usr/bin/env bash
# Fetch the exact Adult.csv used by CTAB-GAN+ (and therefore by the baseline numbers QTabGAN quotes).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
tmp=$(mktemp -d)
git clone -q --depth 1 https://github.com/Team-TUD/CTAB-GAN-Plus.git "$tmp/ctabganplus"
cp "$tmp/ctabganplus/Real_Datasets/Adult.csv" data/Adult.csv
rm -rf "$tmp"
python3 -c "import pandas as pd; d=pd.read_csv('data/Adult.csv'); assert d.shape==(48842,14), d.shape; print('data/Adult.csv ok', d.shape)"
