#!/bin/bash 
echo "validate facenet"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

# directory to load model
md=~
echo "model directory" ${md}

python3 -m facenet.embeddings \
    ${md}/models/facenet/pretrained/20190410-013706 \
    ${ds}/datasets/megaface/megaface_mtcnnaligned_160 \
    --tfrecord ${ds}/datasets/megaface \
    --nrof_folders 0 \
    --use_fixed_image_standardization \
