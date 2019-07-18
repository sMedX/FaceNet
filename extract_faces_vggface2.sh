#!/bin/bash 
echo "align VGGFace2 datasets using extract_faces"

size=160
margin=32
detector=frcnnv3

python facenet/apps/extract_faces.py \
 ~/datasets/vggface2/train/ \
 ~/datasets/vggface2/train_${detector}extracted_${size} \
 --detector ${detector} \
 --image_size ${size} \
 --margin ${margin} \
