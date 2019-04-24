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
    ${ds}/datasets/megaface/megaface_mtcnnaligned_160 \
    --nrof_folders 0 \
    --distance_metric 0 \
    --subtract_mean \
    --use_fixed_image_standardization \
    --false_positive_dir ${ds}/megaface_false_positive_pairs \
    --false_negative_dir ${ds}/megaface_false_negative_pairs
