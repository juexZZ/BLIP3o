#!/bin/bash
cd /group-volume/juexiaozhang/
source ~/.bashrc
conda activate blip3o
source set_path.sh
cd BLIP3o

python data_download.py