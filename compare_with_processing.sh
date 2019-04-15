#!/bin/bash 
echo "validate on lfw"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

# directory to load model
md=~
echo "model directory" ${md}

#model_name="20190327-175738"
model_name="20190410-013706"
echo "model name" ${model_name}

python3 -m facenet.compare_with_processing \
    ${md}/models/facenet/pretrained/${model_name} \
    ${ds}/datasets/lfw/Aaron_Peirsol/Aaron_Peirsol_0001.jpg ${ds}/datasets/lfw/Aaron_Peirsol/Aaron_Peirsol_0002.jpg ${ds}/datasets/lfw/Aaron_Peirsol/Aaron_Peirsol_0003.jpg ${ds}/datasets/lfw/Aaron_Peirsol/Aaron_Peirsol_0004.jpg \
    --rotation 5 \

