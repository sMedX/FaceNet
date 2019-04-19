#!/bin/bash 
echo "validate facenet"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

# directory to load model
md=~
echo "model directory" ${md}

python3 -m facenet.validate \
    ${md}/models/facenet/pretrained/20190327-175738 \
    ${ds}/datasets/megaface/megaface_mtcnnalign_160 \
    --nrof_folders 0 \
    --distance_metric 0 \
    --subtract_mean \
    --use_fixed_image_standardization
