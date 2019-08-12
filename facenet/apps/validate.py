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

import sys
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import math
import numpy as np
import argparse
import pathlib as plib

from facenet import dataset, h5utils
from facenet.statistics import Validation
from facenet import facenet

from facenet.config import DefaultConfig
config = DefaultConfig()


def main(args):

    # Get the paths for the corresponding images
    print('dataset', args.dir)
    print('h5file ', args.h5file)

    dbase = dataset.dataset(args.dir, args.nrof_classes, h5file=args.h5file)
    nrof_images = sum(len(x) for x in dbase)

    print('number of classes', len(dbase))
    print('number of images', nrof_images)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
 
            nrof_preprocess_threads = 4
            image_size = (args.image_size, args.image_size)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int32, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
     
            # load the model to validate
            if args.model == 'default':
                args.model = config.model
            else:
                args.model = plib.Path(args.model).expanduser()
            print('Pre-trained model: {}'.format(args.model))

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

    # Run forward pass to calculate embeddings
    print('Running forward pass on images')

    # Enqueue one epoch of image paths and labels
    nrof_embeddings = sum(len(x) for x in dbase)

    nrof_flips = 2 if args.use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips

    ds_files = []
    for cls in dbase:
        ds_files += cls.image_paths

    labels_array = np.expand_dims(np.arange(0, nrof_images), 1)
    image_paths_array = np.expand_dims(np.repeat(np.array(ds_files), nrof_flips), 1)
    control_array = np.zeros_like(labels_array, np.int32)

    if args.image_standardization:
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION

    if args.use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*facenet.FLIP

    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
    embedding_size = int(embeddings.get_shape()[1])

    batch_size = args.batch_size
    nrof_batches = math.ceil(nrof_images/args.batch_size)

    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))

    for i in range(nrof_batches):
        print('\rEvaluate embeddings {}/{}'.format(i, nrof_batches), end='')

        if (i+1) == nrof_batches:
            batch_size = nrof_images % args.batch_size
            if batch_size == 0:
                batch_size = args.batch_size

        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb

    print('')

    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))

    if args.use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:, :embedding_size] = emb_array[0::2, :]
        embeddings[:, embedding_size:] = emb_array[1::2, :]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images)), \
        'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'

    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)

    stats = Validation(thresholds, embeddings, dbase,
                       far_target=1e-3,
                       nrof_folds=args.nrof_folds,
                       distance_metric=args.distance_metric,
                       subtract_mean=args.subtract_mean)

    stats.print()
    stats.write_report(args.report, dbase, args)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
        default='default')
    parser.add_argument('dir', type=str,
        help='Path to the data directory containing aligned face patches.')
    parser.add_argument('--report', type=str,
        help='File to write statistical report.', default='report.txt')
    parser.add_argument('--nrof_classes', type=int,
        help='Number of classes to validate model.', default=0)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch in the test set.', default=100)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=config.image_size)
    parser.add_argument('--nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--distance_metric', type=int,
        help='Distance metric  0:euclidian, 1:cosine similarity.', default=config.distance_metric)
    parser.add_argument('--use_flipped_images',
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    parser.add_argument('--subtract_mean',
        help='Subtract feature mean before calculating distance.', action='store_true')
    parser.add_argument('--image_standardization', type=bool,
        help='Performs standardization of images: 0 - per image standardization, 1 - fixed standardisation.',
        default=config.image_standardization)
    parser.add_argument('--h5file', type=str,
        help='Path to h5 file with information about valid images.', default=None)
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    main(parse_arguments(sys.argv))
