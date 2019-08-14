#!/bin/bash 
echo "evaluate embeddings"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

# directory to load model
md=~
echo "model directory" ${md}

python3 -m facenet.apps.embeddings \
    ${ds}/datasets/megaface/megaface_frcnnv3extracted_160
