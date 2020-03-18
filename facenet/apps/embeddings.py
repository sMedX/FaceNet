"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2019 SMedX

import click
import time
from pathlib import Path
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from facenet import dataset, ioutils, utils, config, facenet

DefaultConfig = config.DefaultConfig()


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options for the application.')
@click.option('--model', default=None, type=Path,
              help='Could be either a directory containing the meta and ckpt files or a model protobuf (.pb) file')
def main(**args_):
    args = config.YAMLConfig(args_['config'])

    if args_['model'] is not None:
        args.model = args_['model']

    if args.model is None:
        args.model = DefaultConfig.model

    if args.tfrecord is None:
        args.tfrecord = Path(args.dataset.path).expanduser()
    else:
        args.tfrecord = Path(args.tfrecord).expanduser()

    args.tfrecord = Path(str(args.tfrecord) + '_' + args.model.stem)
    ioutils.makedirs(args.tfrecord)
    print('tf record files will be saved to directory {}'.format(args.tfrecord))

    # Get the paths for the corresponding images
    dbase = dataset.DBase(args.dataset)
    print(dbase)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
 
            nrof_preprocess_threads = 4
            image_size = (args.image.size, args.image.size)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=dbase.nrof_images,
                                                       dtypes=[tf.string, tf.int32, tf.int32],
                                                       shapes=[(1,), (1,), (1,)],
                                                       shared_name=None, name=None)

            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder,
                                                             labels_placeholder,
                                                             control_placeholder], name='eval_enqueue_op')

            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue,
                                                                     image_size,
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

    # Run forward pass to calculate embeddings
    print('Running forward pass on images')
    embedding_size = int(embeddings.get_shape()[1])

    for cls_index, cls in enumerate(dbase.classes):
        print("============================================")
        print('({}/{}) ({}) {}'.format(cls_index, dbase.nrof_classes, len(cls.files), cls.name))
        nrof_images = len(cls.files)
        labels_array = np.expand_dims(np.arange(0, nrof_images), 1)
        control_array = np.zeros_like(labels_array, np.int32)

        image_paths_array = np.expand_dims(np.array(cls.files), 1)

        if args.image.standardization:
            control_array += np.ones_like(labels_array) * facenet.FIXED_STANDARDIZATION

        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array,
                              labels_placeholder: labels_array,
                              control_placeholder: control_array})

        batch_size = args.batch_size
        nrof_batches = math.ceil(nrof_images / args.batch_size)

        emb_array = np.zeros((nrof_images, embedding_size))
        lab_array = np.zeros((nrof_images,), dtype=np.int64)

        elapsed_time = 0

        for i in range(nrof_batches):
            print('\rEvaluate embeddings {}/{}'.format(i, nrof_batches), end=utils.end(i, nrof_batches))

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

        assert np.array_equal(lab_array, np.arange(nrof_images)), \
            'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'

        # write tf record file
        emb_array = (emb_array.transpose() / np.linalg.norm(emb_array, axis=1)).transpose()

        tf_file = args.tfrecord.joinpath(cls.name + '.tfrecord')
        utils.write_tfrecord(tf_file, cls.files, [cls_index] * nrof_images, emb_array)

        print('  elapsed time: {}'.format(elapsed_time))
        print('time per image: {}'.format(elapsed_time / emb_array.shape[0]))


if __name__ == '__main__':
    main()
