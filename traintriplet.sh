#!/bin/bash 
echo "train triplet loss"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

# output logs and models directories
md=~
echo "output directory" ${md}

python3 -m facenet.train_tripletloss \
    --model_def facenet.models.inception_resnet_v1 \
    --data_dir ${ds}/datasets/vggface2/train_mtcnn_182 \
    --image_size 160 \
    --models_base_dir ${md}/models/facenet \
    --learning_rate_schedule_file learning_rate_schedule_classifier_vggface2.txt \
    --optimizer ADAM \
    --learning_rate -1 \
    --max_nrof_epochs 500 \
    --keep_probability 0.4 \
    --random_flip \
    --weight_decay 5e-4 \
    --embedding_size 512 \
    --lfw_dir ${ds}/datasets/lfw_mtcnnalign_160 \
    --lfw_distance_metric 1 \
    --lfw_use_flipped_images \
    --lfw_subtract_mean \
    --validation_set_split_ratio 0.01 \
    --validate_every_n_epochs 5
