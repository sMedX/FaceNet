#!/bin/bash 
echo "train softmax"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

# output logs and models directories
md=~
echo "output directory" ${md}

python3 -m facenet.apps.train_softmax \
    --model_def facenet.models.inception_resnet_v1 \
    --data_dir ${ds}/datasets/vggface2/test_frcnnv3extracted_160 \
    --models_base_dir ${md}/models/facenet \
    --optimizer ADAM \
    --learning_rate -1 \
    --learning_rate_schedule_file data/learning_rate_schedule_classifier_vggface2.txt \
    --max_nrof_epochs 500 \
    --keep_probability 0.4 \
    --weight_decay 5e-4 \
    --lfw_dir ${ds}/datasets/lfw_mtcnnaligned_160 \
    --validation_set_split_ratio 0.01 \
    --validate_every_n_epochs 5
