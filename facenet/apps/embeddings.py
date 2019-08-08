"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2019 SMedX
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys, os
import tensorflow as tf
import math
import numpy as np
import argparse
from facenet import dataset, utils
from facenet import facenet
from tensorflow.python.ops import data_flow_ops

from facenet.config import DefaultConfig
config = DefaultConfig()


def main(args):

    # Get the paths for the corresponding images
    dbase = dataset.dataset(args.dir, args.nrof_classes)
    nrof_images = sum(len(x) for x in dbase)

    print('dataset', args.dir)
    print('number of classes', len(dbase))
    print('number of images', nrof_images)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
 
            nrof_preprocess_threads = 4
            image_size = (args.image_size, args.image_size)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=nrof_images,
                                                       dtypes=[tf.string, tf.int32, tf.int32],
                                                       shapes=[(1,), (1,), (1,)],
                                                       shared_name=None, name=None)

            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder,
                                                             labels_placeholder,
                                                             control_placeholder], name='eval_enqueue_op')

            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue,
                                                                     image_size,
                                                                     nrof_preprocess_threads,
                                                                     batch_size_placeholder)
     
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(args.model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            evaluate(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                     batch_size_placeholder, control_placeholder, embeddings, label_batch, dbase, args)


def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
             batch_size_placeholder, control_placeholder, embeddings, labels, dbase, args):

    if args.tfrecord is None:
        tf_dir = os.path.expanduser(args.dir)
    else:
        tf_dir = os.path.expanduser(args.tfrecord)

    model = os.path.splitext(os.path.basename(args.model))[0]
    tf_dir = os.path.join(tf_dir, model)
    if not os.path.isdir(tf_dir):
        os.mkdir(tf_dir)
    print('save tf records to dir {}'.format(tf_dir))

    # Run forward pass to calculate embeddings
    print('Running forward pass on images')
    embedding_size = int(embeddings.get_shape()[1])

    for cls_index, cls in enumerate(dbase):
        print("============================================")
        print('({}/{}) ({}) {}'.format(cls_index, len(dbase), len(cls.image_paths), cls.name))
        nrof_images = len(cls.image_paths)
        labels_array = np.expand_dims(np.arange(0, nrof_images), 1)
        control_array = np.zeros_like(labels_array, np.int32)

        image_paths_array = np.expand_dims(np.array(cls.image_paths), 1)

        if args.image_standardization:
            control_array += np.ones_like(labels_array) * facenet.FIXED_STANDARDIZATION

        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array,
                              labels_placeholder: labels_array,
                              control_placeholder: control_array})

        batch_size = args.batch_size
        nrof_batches = math.ceil(nrof_images / args.batch_size)

        emb_array = np.zeros((nrof_images, embedding_size))
        lab_array = np.zeros((nrof_images,), dtype=np.int64)

        for i in range(nrof_batches):
            print('\rEvaluate embeddings {}/{}'.format(i, nrof_batches), end=utils.end(i, nrof_batches))

            if (i+1) == nrof_batches:
                batch_size = nrof_images % args.batch_size
                if batch_size == 0:
                    batch_size = args.batch_size

            feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size}
            emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
            lab_array[lab] = lab
            emb_array[lab, :] = emb

        assert np.array_equal(lab_array, np.arange(nrof_images)), \
            'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'

        # write tf record file
        emb_array = (emb_array.transpose() / np.linalg.norm(emb_array, axis=1)).transpose()

        tf_file = os.path.join(tf_dir, cls.name + '.tfrecord')
        utils.write_tfrecord(tf_file, cls.image_paths, [cls_index]*nrof_images, emb_array)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dir', type=str,
        help='Path to the data directory containing aligned face patches.')
    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
        default=config.model)
    parser.add_argument('--tfrecord', type=str,
        help='Path to save tf record file.', default=None)
    parser.add_argument('--nrof_classes', type=int,
        help='Number of classes to evaluate embeddings.', default=0)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch in the test set.', default=500)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=config.image_size)
    parser.add_argument('--image_standardization', type=bool,
        help='Performs standardization of images: 0 - per image standardization, 1 - fixed standardisation.',
        default=config.image_standardization)
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    main(parse_arguments(sys.argv))
