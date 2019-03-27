#!/bin/bash 
echo "align VGGFace2 datasets using mtcnn"

python3.5m -m facenet.align.align_dataset_mtcnn \
    ~/datasets/vggface2/train/ \
    ~/datasets/vggface2/train_mtcnnalign_182 \
    --gpu_memory_fraction 1.0 \
    --image_size 182 \
    --margin 44

python3.5m -m facenet.align.align_dataset_mtcnn \
    ~/datasets/vggface2/test/ \
    ~/datasets/vggface2/test_mtcnnalign_182 \
    --gpu_memory_fraction 1.0 \
    --image_size 182 \
    --margin 44

