#!/bin/bash 
echo "evaluate embeddings"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

# directory to load model
md=~
echo "model directory" ${md}

python3 -m facenet.embeddings \
    ${md}/models/facenet/pretrained/20190410-013706 \
    ${ds}/datasets/vggface2/train_mtcnnaligned_160 \
    --nrof_folders 0 \
    --tfrecord ${ds}/datasets/vggface2 \
    --use_fixed_image_standardization \
