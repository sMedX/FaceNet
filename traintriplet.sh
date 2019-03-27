#!/bin/bash 
echo "train triplet loss"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

# output logs and models directories
md=~
echo "output directory" ${md}

python3.5m -m facenet.train_tripletloss \
    --model_def facenet.models.inception_resnet_v1 \
    --data_dir ${ds}/datasets/vggface2/train_mtcnn_182 \
    --image_size 160 \
    --lfw_dir ${ds}/datasets/lfw_mtcnnalign_160 \
    --logs_base_dir ${md}/facenet/triplet/logs \
    --models_base_dir ${md}/facenet/triplet/models \
    --optimizer RMSPROP \
    --learning_rate 0.01 \
    --weight_decay 1e-4 \
    --max_nrof_epochs 500