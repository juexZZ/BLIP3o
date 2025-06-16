#!/bin/bash
cd /group-volume/juexiaozhang/
source ~/.bashrc
conda activate blip3o
source set_path.sh
cd BLIP3o

# echo "embedding for chunk" $1 $2
# python retrieval_chunk.py \
#     --data_name coco-val \
#     --out_dir /group-volume/juexiaozhang/BLIP3o/embedding_coco_val_consqchunk \
#     --index $1 \
#     --n_chunks $2 \
#     --seed 42 \

# python retrieval_chunk.py \
#     --data_name coco-val \
#     --out_dir /group-volume/juexiaozhang/BLIP3o/embedding_coco_val_consqchunk \
#     --index $1 \
#     --n_chunks $2 \
#     --seed 123 \

# python retrieval_chunk.py \
#     --data_name coco-val \
#     --out_dir /group-volume/juexiaozhang/BLIP3o/embedding_coco_val_consqchunk \
#     --index $1 \
#     --n_chunks $2 \
#     --seed 456 \

# python retrieval_chunk.py \
#     --data_name coco-val \
#     --out_dir /group-volume/juexiaozhang/BLIP3o/embedding_coco_val_consqchunk \
#     --index $1 \
#     --n_chunks $2 \
#     --seed 789 \

python retrieval_chunk.py \
    --data_name coco-val \
    --out_dir /group-volume/juexiaozhang/BLIP3o/embedding_coco_val_consqchunk \
    --index $1 \
    --n_chunks $2 \
    --seed 101112 \