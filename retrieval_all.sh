#!/bin/bash
cd /group-volume/juexiaozhang/
source ~/.bashrc
conda activate blip3o
source set_path.sh
cd BLIP3o

# python test_retrieval.py coco-val
python test_retrieval_evaclip.py coco-val