# coding:utf-8
"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
# 
# Copyright (c) 2020 sMedX
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

import click
import time
import math
import numpy as np
import importlib
from tqdm import tqdm
from functools import partial
from pathlib import Path
import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim

from facenet import ioutils, dataset, statistics, config, h5utils, facenet, tfutils


def load_images(path, image_size):
    contents = tf.io.read_file(path)
    image = tf.image.decode_image(contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
    image = (tf.cast(image, tf.float32) - 127.5) / 128
    return image


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options of the application.')
@click.option('--learning_rate', default=None, type=float,
              help='Learning rate value')
def main(**args_):
    start_time = time.monotonic()
    args = config.TrainOptions(args_, subdir=config.subdir())

    # import network
    print('import model {}'.format(args.model.module))
    network = importlib.import_module(args.model.module)

    # ------------------------------------------------------------------------------------------------------------------
    dbase = dataset.DBase(args.dataset)
    dbase, dbase_val = dbase.random_split(args.validate)
    ioutils.write_text_log(args.txtfile, str(dbase))
    ioutils.write_text_log(args.txtfile, str(dbase_val))
    print('train dbase:', dbase)
    print('validate dbase:', dbase_val)

    dbase_emb = dataset.DBase(args.validate.dataset)
    ioutils.write_text_log(args.txtfile, str(dbase_emb))
    print(dbase_emb)

    # ------------------------------------------------------------------------------------------------------------------
    tf.reset_default_graph()
    tf.Graph().as_default()

    with tf.Graph().as_default():
        map_func = partial(load_images, image_size=args.image.size)
        ds_validate = {
            'validate': facenet.make_validate_dataset(dbase_val, map_func, args),
            'embedding': facenet.make_validate_dataset(dbase_emb, map_func, args)
        }

        global_step = tf.Variable(0, trainable=False, name='global_step')

        placeholders = facenet.Placeholders(args.image.size)

        print('Building training graph')
        prelogits, _ = network.inference(tf.identity(placeholders.image_batch, 'input'),
                                         config=args.model.config,
                                         phase_train=placeholders.phase_train)

        logits = slim.fully_connected(prelogits, dbase.nrof_classes, activation_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      weights_regularizer=slim.l2_regularizer(args.model.config.weight_decay),
                                      scope='Logits', reuse=False)

        embedding = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embedding')

        # Norm for the prelogits
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=args.loss.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.loss.prelogits_norm_factor)

        # Add center loss
        prelogits_center_loss, _ = facenet.center_loss(prelogits, placeholders.label_batch, args.loss.center_alfa,
                                                       dbase.nrof_classes)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.loss.center_factor)

        # define learning rate tensor
        learning_rate = tf.train.exponential_decay(placeholders.learning_rate, global_step,
                                                   args.train.learning_rate.decay_epochs * args.train.epoch.size,
                                                   args.train.learning_rate.decay_factor, staircase=True)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=placeholders.label_batch,
                                                                       logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(placeholders.label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train_op(args.train, total_loss, global_step, learning_rate, tf.global_variables())

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        summary_writer = tf.summary.FileWriter(args.logs, sess.graph)

        with sess.as_default():
            facenet.restore_checkpoint(saver, sess, args.model.checkpoint)
            tf.train.global_step(sess, global_step)
            sess.run(global_step.initializer)

            tensor_dict = {
                'train_op': train_op,
                'summary_op': summary_op,
                'tensor_op': {
                    'accuracy': accuracy,
                    'loss': total_loss,
                    'xent': cross_entropy_mean,
                    'center_loss': prelogits_center_loss,
                    'prelogits_norm': prelogits_norm,
                    'learning_rate': learning_rate
                }
            }

            train_summary = facenet.Summary(summary_writer, args.h5file, tag='train')

            val_tensor_dict = {
                'embedding': embedding,
                'tensor_op': {
                    'accuracy': accuracy,
                    'loss': total_loss,
                    'xent': cross_entropy_mean
                }
            }

            val_summary = facenet.Summary(summary_writer, args.h5file, tag='validate')

            # Training and validation loop
            for epoch in range(args.train.epoch.nrof_epochs):
                info = '(model {}, epoch [{}/{}])'.format(args.model.path.stem, epoch+1, args.train.epoch.nrof_epochs)

                # train for one epoch
                ds_train = facenet.make_train_dataset(dbase, map_func, args)
                train(args, sess, epoch, tensor_dict, train_summary, info, placeholders, ds_train)

                # save variables and the meta graph if it doesn't exist already
                tfutils.save_variables_and_metagraph(sess, saver, args.model.path, epoch)

                # perform validation
                epoch1 = epoch + 1
                if not epoch1 % args.validate.every_n_epochs or epoch1 == args.train.epoch.nrof_epochs:
                    validate(sess, ds_validate['validate'], placeholders, val_tensor_dict, val_summary, info)

                    # perform face-to-face validation
                    tfutils.save_freeze_graph(model_dir=args.model.path, suffix='-{}'.format(epoch))
                    embeddings, labels = facenet.evaluate_embeddings(sess, embedding, ds_validate['embedding'], placeholders)

                    validation = statistics.FaceToFaceValidation(embeddings, labels, args.validate.validate)

                    ioutils.write_text_log(args.txtfile, str(validation))
                    h5utils.write_dict(args.h5file, validation.dict, group='validate')
                    for key, value in validation.dict.items():
                        val_summary.write_tf_summary(value, tag='{}_{}'.format('validate', key))

                    print(validation)

    ioutils.write_elapsed_time(args.h5file, start_time)
    ioutils.write_elapsed_time(args.txtfile, start_time)

    print('Statistics have been saved to the h5 file: {}'.format(args.h5file))
    print('Logs have been saved to the directory: {}'.format(args.logs))
    print('Model has been saved to the directory: {}'.format(args.model.path))

    return args.model.path


def train(args, sess, epoch, tensor_dict, summary, info, placeholders, ds):
    print('\nRunning training', info)
    start_time = time.monotonic()

    learning_rate = facenet.learning_rate_value(epoch, args.train.learning_rate)
    if not learning_rate:
        return False

    outputs = {key: [] for key in tensor_dict['tensor_op'].keys()}

    # nrof_batches = sess.run(tf.data.experimental.cardinality(ds))
    nrof_batches = args.train.epoch.size

    iterator = ds.make_one_shot_iterator().get_next()

    with tqdm(total=nrof_batches) as bar:
        for batch_number in range(nrof_batches):
            image_batch, label_batch = sess.run(iterator)

            feed_dict = placeholders.train_feed_dict(image_batch, label_batch, learning_rate)
            output = sess.run(tensor_dict, feed_dict=feed_dict)

            for key, value in output['tensor_op'].items():
                outputs[key].append(value)

            summary.write_tf_summary(output)

            bar.set_postfix_str(summary.get_info_str(output))
            bar.update()

    summary.write_h5_summary(outputs)
    summary.write_elapsed_time(time.monotonic() - start_time)

    return True


def validate(sess, dataset, placeholders, tensor_dict, summary, info):
    print('\nRunning forward pass on validation set', info)
    start_time = time.monotonic()

    outputs = {key: 0 for key in tensor_dict['tensor_op'].keys()}

    # embeddings = np.zeros((0, args.model.config.embedding_size))
    nrof_batches = sess.run(tf.data.experimental.cardinality(dataset))
    iterator = dataset.make_one_shot_iterator().get_next()

    with tqdm(total=nrof_batches) as bar:
        for i in range(nrof_batches):
            image_batch, label_batch = sess.run(iterator)
            output = sess.run(tensor_dict,
                              feed_dict=placeholders.validate_feed_dict(image_batch, label_batch))

            for key, value in output['tensor_op'].items():
                outputs[key] = (outputs[key]*i + value)/(i+1)

            bar.set_postfix_str(summary.get_info_str(outputs))
            bar.update()

    summary.write_tf_summary(outputs)
    summary.write_h5_summary(outputs)
    summary.write_elapsed_time(time.monotonic() - start_time)


if __name__ == '__main__':
    main()

