#!/bin/bash 
echo "evaluate embeddings"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

# directory to load model
md=~
echo "model directory" ${md}

python3 -m facenet.embeddings \
    ${ds}/datasets/vggface2/train_frcnnv3extracted_160 \
    --nrof_folders 0 \
    --tfrecord ${ds}/datasets/vggface2
