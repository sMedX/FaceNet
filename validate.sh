#!/bin/bash 
echo "validate facenet"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

# directory to load model
md=~
echo "model directory" ${md}

python3 -m facenet.apps.validate \
    ${ds}/datasets/vggface2/test_frcnnv3extracted_160 \
    --model default \
    --use_fixed_image_standardization \
