#!/bin/bash 
echo "freeze graph"

# model directory
md=~
echo "model directory" ${md}

python3 -m facenet.apps.freeze_graph \
  ${md}/models/facenet/20190721-142131
