"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2019 SMedX

import click
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import math
import time
import numpy as np
import pathlib
from facenet import dataset, config, facenet
from facenet.statistics import Validation

DefaultConfig = config.DefaultConfig()


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=pathlib.Path,
              help='Path to yaml config file with used options for the application.')
def main(**args_):
    args = config.YAMLConfig(args_['config'])
    if args.model is None:
        args.model = DefaultConfig.model
        args.image_size = DefaultConfig.image_size

    # Get the paths for the corresponding images
    dbase = dataset.DBase(args.dataset, nrof_classes=args.dataset.nrof_classes)
    print(dbase)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            image_size = (args.image_size, args.image_size)

            eval_input_queue = data_flow_ops.FIFOQueue(capacity=dbase.nrof_images,
                                                       dtypes=[tf.string, tf.int32, tf.int32],
                                                       shapes=[(1,), (1,), (1,)],
                                                       shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')

            nrof_preprocess_threads = 4
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
     
            # load the model to validate
            args.model = pathlib.Path(args.model).expanduser()
            print('model: {}'.format(args.model))

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
    nrof_embeddings = dbase.nrof_images

    nrof_flips = 2 if args.image.use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips

    labels_array = np.expand_dims(np.arange(0, nrof_images), 1)
    image_paths_array = np.expand_dims(np.repeat(np.array(dbase.files), nrof_flips), 1)
    control_array = np.zeros_like(labels_array, np.int32)

    if args.image.standardization:
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION

    if args.image.use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*facenet.FLIP

    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
    embedding_size = int(embeddings.get_shape()[1])

    batch_size = args.batch_size
    nrof_batches = math.ceil(nrof_images/args.batch_size)

    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    elapsed_time = 0

    for i in range(nrof_batches):
        print('\rEvaluate embeddings {}/{}'.format(i, nrof_batches), end='')

        if (i+1) == nrof_batches:
            batch_size = nrof_images % args.batch_size
            if batch_size == 0:
                batch_size = args.batch_size

        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size}

        start = time.monotonic()
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        elapsed_time += time.monotonic() - start
        lab_array[lab] = lab
        emb_array[lab, :] = emb

    print('')

    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))

    if args.image.use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:, :embedding_size] = emb_array[0::2, :]
        embeddings[:, embedding_size:] = emb_array[1::2, :]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images)), \
        'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'

    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)

    stats = Validation(thresholds, embeddings, dbase.labels,
                       far_target=1e-3,
                       nrof_folds=args.validation.nrof_folds,
                       metric=args.validation.metric,
                       subtract_mean=args.validation.subtract_mean)
    stats.write_report(elapsed_time, args, file=args.report, dbase_info=dbase.__repr__())


if __name__ == '__main__':
    main()
