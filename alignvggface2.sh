#!/bin/bash 
echo "align VGGFace2 datasets using mtcnn"

size=160
margin=32

python3.5m -m facenet.align.align_dataset_mtcnn \
    ~/datasets/vggface2/train/ \
    ~/datasets/vggface2/train_mtcnnaligned_${size} \
    --image_size ${size} \
    --margin ${margin}

python3.5m -m facenet.align.align_dataset_mtcnn \
    ~/datasets/vggface2/test/ \
    ~/datasets/vggface2/test_mtcnnaligned_${size} \
    --image_size ${size} \
    --margin ${margin}

