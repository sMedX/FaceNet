#!/bin/bash 
echo "write false examples"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

python3 -m facenet.false_examples \
    ${ds}/datasets/vggface2/test_mtcnnaligned_160 \
    ${ds}/datasets/vggface2/test_mtcnnaligned_16020190410-013706.tfrecord \
    --threshold 0.82000 \
    --false_positive_dir ${ds}/datasets/vggface2/vggface2_test_false_positive \
    --false_negative_dir ${ds}/datasets/vggface2/vggface2_test_false_negative