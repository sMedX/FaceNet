#!/bin/bash 
echo "align datasets using mtcnn"

python3.5m -m facenet.align.align_dataset_mtcnn \
    ~/datasets/lfw/ \
    ~/datasets/lfw_mtcnnalign_160 \
    --gpu_memory_fraction 1.0 \
    --image_size 160 \
    --margin 32 \
    --random_order \
