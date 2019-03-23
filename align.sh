#!/bin/bash 
echo "align datasets using mtcnn"

for N in {1..4}
do
python3.5m -m facenet.align.align_dataset_mtcnn \
    ~/datasets/VGGFace2/train/ \
    ~/datasets/VGGFace2/train_mtcnn_182 \
    --gpu_memory_fraction 0.25 \
    --image_size 182 \
    --margin 44
done 
wait

