#!/bin/bash 
echo "validate on lfw"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

# directory to load model
md=~
echo "model directory" ${md}

python3 -m facenet.apps.validate_on_lfw \
    ${ds}/datasets/lfw_mtcnnaligned_160 \
    --distance_metric 0 \
    --subtract_mean \
    --use_fixed_image_standardization
