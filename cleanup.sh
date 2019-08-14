#!/bin/bash 
echo "clean up dataset"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

python3 -m facenet.apps.find_similar_faces \
    ${ds}/datasets/megaface/megaface_frcnnv3extracted_160_20190727-080213 \

python3 -m facenet.apps.find_outlier_faces \
    ${ds}/datasets/megaface/megaface_frcnnv3extracted_160_20190727-080213 \
