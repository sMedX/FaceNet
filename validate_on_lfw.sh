#!/bin/bash 
echo "validate on lfw"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

# directory to load model
md=~
echo "model directory" ${md}

python3 -m facenet.validate_on_lfw \
    ${ds}/datasets/lfw_mtcnnalign_160 \
    ${md}/models/facenet/pretrained/20190327-175738 \
    --distance_metric 1 \
    --use_flipped_images \
    --subtract_mean \
    --use_fixed_image_standardization
