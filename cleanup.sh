#!/bin/bash 
echo "write false examples"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

python3 -m facenet.apps.cleanup \
    ${ds}/datasets/megaface/megaface_frcnnv3extracted_160_20190727-080213 \
