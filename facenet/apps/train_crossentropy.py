# coding:utf-8
"""Training a face recognizer with TensorFlow using binary cross entropy loss
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
import importlib
from tqdm import tqdm
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf

from facenet import ioutils, dataset, statistics, config, h5utils, facenet, tfutils


class Placeholders:
    def __init__(self):
        self.image_batch = tf.placeholder(tf.uint8, shape=[None, None, None, 3], name='image_batch')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    def train_feed_dict(self, image_batch, learning_rate):
        return {
            self.image_batch: image_batch,
            self.learning_rate: learning_rate,
            self.phase_train: True,
            self.batch_size: image_batch.shape[0]
        }

    def validate_feed_dict(self, image_batch):
        return {
            self.image_batch: image_batch,
            self.phase_train: False,
            self.batch_size: image_batch.shape[0]
        }


# binary cross entropy loss
def binary_cross_entropy_loss(embeddings, options):
    alpha = tf.Variable(initial_value=10., dtype=tf.float32, name='alpha')
    threshold = tf.Variable(initial_value=1., dtype=tf.float32, name='threshold')
    loss_vars = {'alpha': alpha, 'threshold': threshold}

    # define upper-triangle indices
    batch_size = options.nrof_classes_per_batch * options.nrof_examples_per_class
    triu_indices = [(i, k) for i, k in zip(*np.triu_indices(batch_size, k=1))]

    # compute labels for embeddings
    labels = []
    for i, k in triu_indices:
        if i // options.nrof_examples_per_class == k // options.nrof_examples_per_class:
            # label 1 means inner class distance
            labels.append(1)
        else:
            # label 0 means across class distance
            labels.append(0)

    pos_weight = len(labels)/sum(labels) - 1

    # initialize cross entropy loss
    distances = 2 * (1 - embeddings @ tf.transpose(embeddings))
    distances = tf.gather_nd(distances, triu_indices)

    logits = tf.multiply(alpha, tf.subtract(threshold, distances))
    labels = tf.constant(labels, dtype=logits.dtype)

    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels, logits, pos_weight)
    loss = tf.reduce_mean(cross_entropy)

    return loss, loss_vars


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options of the application.')
def main(**options):
    start_time = time.monotonic()
    options = config.TrainOptions(options, subdir=config.subdir())

    # import network
    print('import model {}'.format(options.model.module))
    network = importlib.import_module(options.model.module)

    # ------------------------------------------------------------------------------------------------------------------
    dbase = dataset.DBase(options.dataset)
    ioutils.write_text_log(options.txtfile, str(dbase))
    print('train dbase:', dbase)

    dbase_val = dataset.DBase(options.validate.dataset)
    ioutils.write_text_log(options.txtfile, str(dbase_val))
    print('validate dbase', dbase_val)

    load_images = partial(facenet.load_images, options=options.image)

    # build input pipeline for binary crossentropy loss
    image_batch, label_batch = facenet.binary_cross_entropy_input_pipeline(dbase, options)

    val_ds = facenet.make_validate_dataset(dbase_val, load_images, options)
    val_iter = val_ds.make_initializable_iterator()
    val_elem = val_iter.get_next()

    # ------------------------------------------------------------------------------------------------------------------
    print('Building training graph')

    global_step = tf.Variable(0, trainable=False, name='global_step')

    placeholders = Placeholders()

    prelogits, _ = network.inference(facenet.image_processing(placeholders.image_batch, options.image),
                                     config=options.model.config,
                                     phase_train=placeholders.phase_train)
    embedding = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embedding')

    cross_entropy, loss_vars = binary_cross_entropy_loss(embedding, options)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([cross_entropy] + regularization_losses, name='loss')

    learning_rate = tf.train.exponential_decay(placeholders.learning_rate, global_step,
                                               options.train.learning_rate.decay_epochs * options.train.epoch.size,
                                               options.train.learning_rate.decay_factor, staircase=True)

    train_op = facenet.train_op(options.train, loss, global_step, learning_rate, tf.global_variables())

    # Create a saver
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Start running operations on the Graph.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=options.gpu_memory_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        summary_writer = tf.summary.FileWriter(options.logs, sess.graph)

        tfutils.restore_checkpoint(saver, sess, options.model.checkpoint)
        tf.train.global_step(sess, global_step)
        sess.run(global_step.initializer)

        tensor_ops = {
            'train_op': train_op,
            'summary_op': summary_op,
            'tensor_op': {
                'loss': loss,
                'learning_rate': learning_rate,
                'alpha': loss_vars['alpha'],
                'threshold': loss_vars['threshold']
            }
        }

        summary = {
            'train': facenet.Summary(summary_writer, options.h5file, tag='train'),
            'validate': facenet.Summary(summary_writer, options.h5file, tag='validate')
        }

        # Training and validation loop
        for epoch in range(options.train.epoch.nrof_epochs):
            info = '(model {}, epoch [{}/{}])'.format(options.model.path.stem, epoch+1, options.train.epoch.nrof_epochs)

            # train for one epoch
            train(options, sess, placeholders, epoch, tensor_ops, summary['train'], image_batch, info)

            # save variables and the meta graph if it doesn't exist already
            tfutils.save_variables_and_metagraph(sess, saver, options.model.path, epoch)

            # perform validation
            epoch1 = epoch + 1
            if epoch1 % options.validate.every_n_epochs == 0 or epoch1 == options.train.epoch.nrof_epochs:
                embeddings, labels = facenet.evaluate_embeddings(sess, embedding, placeholders,
                                                                 val_ds, val_iter, val_elem, info)

                validation = statistics.FaceToFaceValidation(embeddings, labels, options.validate.validate, info=info)

                ioutils.write_text_log(options.txtfile, str(validation))
                h5utils.write_dict(options.h5file, validation.dict, group='validate')

                for key, value in validation.dict.items():
                    summary['validate'].write_tf_summary(value, tag='{}_{}'.format('validate', key))

                print(validation)

    tfutils.save_freeze_graph(model_dir=options.model.path, optimize=True)

    ioutils.write_elapsed_time(options.h5file, start_time)
    ioutils.write_elapsed_time(options.txtfile, start_time)

    print('Statistics have been saved to the h5 file: {}'.format(options.h5file))
    print('Logs have been saved to the directory: {}'.format(options.logs))
    print('Model has been saved to the directory: {}'.format(options.model.path))

    return options.model.path


def train(args, sess, placeholders, epoch, tensor_dict, summary, image_batch, info):
    print('\nRunning training', info)
    start_time = time.monotonic()

    learning_rate = facenet.learning_rate_value(epoch, args.train.learning_rate)
    if not learning_rate:
        return False

    outputs = {key: [] for key in tensor_dict['tensor_op'].keys()}

    epoch_size = args.train.epoch.size

    with tqdm(total=epoch_size) as bar:
        for batch_number in range(epoch_size):
            image_batch_np = sess.run(image_batch)

            feed_dict = placeholders.train_feed_dict(image_batch_np, learning_rate)
            output = sess.run(tensor_dict, feed_dict=feed_dict)

            for key, value in output['tensor_op'].items():
                outputs[key].append(value)

            summary.write_tf_summary(output)

            bar.set_postfix_str(summary.get_info_str(output))
            bar.update()

    summary.write_h5_summary(outputs)
    summary.write_elapsed_time(time.monotonic() - start_time)

    return True


def validate(sess, placeholders, dataset, iterator, batch, tensor_dict, summary, info):
    print('\nRunning forward pass on validation set', info)
    start_time = time.monotonic()

    outputs = {key: 0 for key in tensor_dict['tensor_op'].keys()}

    # embeddings = np.zeros((0, args.model.config.embedding_size))
    nrof_batches = sess.run(tf.data.experimental.cardinality(dataset))
    sess.run(iterator.initializer)

    with tqdm(total=nrof_batches) as bar:
        for i in range(nrof_batches):
            image_batch, label_batch = sess.run(batch)
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

