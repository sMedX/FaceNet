# coding:utf-8
"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
# 
# Copyright (c) 2019 sMedX
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
import numpy as np
import importlib
from tqdm import tqdm
from pathlib import Path
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

from facenet import ioutils, h5utils, dataset, statistics, config, facenet


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options of the application.')
@click.option('--learning_rate', default=None, type=float,
              help='Learning rate value')
def main(**args_):
    start_time = time.monotonic()

    args = config.TrainOptions(args_, subdir=config.subdir())

    # import network
    print('import model \'{}\''.format(args.model.module))
    network = importlib.import_module(args.model.module)

    dbase = dataset.DBase(args.dataset)
    train_dbase, val_dbase = dbase.random_split(args.validate.dataset_split_ratio)
    print('train dbase:', train_dbase)
    print('validate dbase:', val_dbase)

    tf.reset_default_graph()
    tf.Graph().as_default()

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Create a queue that produces indices into the image_list and label_list
        labels = ops.convert_to_tensor(train_dbase.labels, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None, shuffle=True, seed=None, capacity=32)
        index_dequeue_op = index_queue.dequeue_many(args.batch_size * args.train.epoch.size, 'index_dequeue')

        placeholders = facenet.Placeholders()
        placeholders.batch_size = tf.placeholder(tf.int32, name='batch_size')
        placeholders.phase_train = tf.placeholder(tf.bool, name='phase_train')
        placeholders.files = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
        placeholders.labels = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
        placeholders.control = tf.placeholder(tf.int32, shape=(None, 1), name='control')
        placeholders.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        input_queue = data_flow_ops.FIFOQueue(capacity=train_dbase.nrof_images,
                                              dtypes=[tf.string, tf.int32, tf.int32],
                                              shapes=[(1,), (1,), (1,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([placeholders.files, placeholders.labels, placeholders.control],
                                              name='enqueue_op')

        image_size = (args.image.size, args.image.size)
        image_batch, label_batch = facenet.create_input_pipeline(input_queue, image_size, placeholders.batch_size)

        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        # Build the inference graph
        print('Building training graph')
        prelogits, _ = network.inference(image_batch,
                                         config=args.model.config,
                                         phase_train=placeholders.phase_train)

        logits = slim.fully_connected(prelogits, train_dbase.nrof_classes, activation_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      weights_regularizer=slim.l2_regularizer(args.model.config.weight_decay),
                                      scope='Logits', reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Norm for the prelogits
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=args.loss.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.loss.prelogits_norm_factor)

        # Add center loss
        prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.loss.center_alfa,
                                                       train_dbase.nrof_classes)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.loss.center_factor)

        # define learning rate tensor
        learning_rate = tf.train.exponential_decay(placeholders.learning_rate, global_step,
                                                   args.train.learning_rate.decay_epochs * args.train.epoch.size,
                                                   args.train.learning_rate.decay_factor, staircase=True)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch,
                                                                       logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
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
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            tf.compat.v1.train.global_step(sess, global_step)
            sess.run(global_step.initializer)

            facenet.restore_checkpoint(saver, sess, args.model.checkpoint)

            nrof_steps = args.train.epoch.nrof_epochs * args.train.epoch.size
            train_tensor_op = facenet.Tensors({'accuracy': accuracy,
                                               'loss': total_loss,
                                               'xent': cross_entropy_mean,
                                               'center_loss': prelogits_center_loss,
                                               'prelogits_norm': prelogits_norm,
                                               'learning_rate': learning_rate}, nrof_steps,
                                              train_op, summary_op, global_step, summary_writer, tag='train')

            nrof_val_steps = math.ceil(args.train.epoch.nrof_epochs / args.validate.every_n_epochs)
            val_tensor_op = facenet.Tensors({'accuracy': accuracy,
                                             'loss': total_loss,
                                             'xent': cross_entropy_mean}, nrof_val_steps,
                                            summary_writer=summary_writer, tag='validate')

            # Training and validation loop
            for epoch in range(args.train.epoch.nrof_epochs):
                # train for one epoch
                train(args, sess, epoch, train_dbase, index_dequeue_op, enqueue_op, placeholders, train_tensor_op)

                # perform validation
                if args.validate:
                    epoch1 = epoch + 1
                    if epoch1 % args.validate.every_n_epochs == 0 or epoch1 == args.train.epoch.nrof_epochs:
                        validate(args, sess, epoch, val_dbase, enqueue_op, placeholders, val_tensor_op)
                        h5utils.write_dict(args.h5logs, val_tensor_op.stats)

                # save variables and the meta graph if it doesn't exist already
                facenet.save_variables_and_metagraph(sess, saver, args.model.path, epoch)

                # save statistics to h5 file
                h5utils.write_dict(args.h5logs, train_tensor_op.stats)

    facenet.save_freeze_graph(model_dir=args.model.path)

    # perform validation
    if args.validation:
        config_ = args.validation
        dbase = dataset.DBase(config_.dataset)
        print(dbase)

        emb = facenet.Embeddings(dbase, config_, model=args.model.path)
        print(emb)

        validation = statistics.Validation(emb.embeddings, dbase.labels, config_.validation)
        validation.write_report(path=args.model.path, info=(dbase.__repr__(), emb.__repr__()))
        print(validation)

        ioutils.elapsed_time(validation.report_file, start_time=start_time)

    print('Model has been saved to the directory: {}'.format(args.model.path))
    print('Logs have been saved to the directory: {}'.format(args.logs))
    return args.model.path


def train(args, sess, epoch, dbase, index_dequeue_op, enqueue_op, placeholders, tensors):
    print('\nRunning training', flush=True)

    learning_rate = facenet.learning_rate_value(epoch, args.train.learning_rate)
    if not learning_rate:
        return False

    index_epoch = sess.run(index_dequeue_op)
    image_epoch = np.array(dbase.files)[index_epoch]
    label_epoch = np.array(dbase.labels)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)

    control_value = (facenet.RANDOM_ROTATE * args.image.random_rotate +
                     facenet.RANDOM_CROP * args.image.random_crop +
                     facenet.RANDOM_FLIP * args.image.random_flip +
                     facenet.FIXED_STANDARDIZATION * args.image.standardization)

    control_array = np.ones_like(labels_array) * control_value
    sess.run(enqueue_op, {placeholders.files: image_paths_array,
                          placeholders.labels: labels_array,
                          placeholders.control: control_array})

    feed_dict = placeholders.train_feed_dict(learning_rate, True, args.batch_size)
    nrof_batches = args.train.epoch.size

    start_time = time.monotonic()

    with tqdm(total=nrof_batches) as bar:
        for batch_number in range(nrof_batches):
            output = sess.run(tensors.tensors, feed_dict=feed_dict)
            tensors.set_output(output)

            # prelogits_hist = np.minimum(np.abs(output['prelogits']), args.loss.prelogits_hist_max)
            # stat['prelogits_hist'][epoch, :] += np.histogram(prelogits_hist, bins=1000, range=(0.0, args.loss.prelogits_hist_max))[0]

            info = '[{}/{}] '.format(epoch + 1, args.train.epoch.nrof_epochs)
            bar.set_postfix_str(info + tensors.get_info_str(output))
            bar.update()

    elapsed_time = time.monotonic() - start_time
    tensors.set_elapsed_time(elapsed_time)

    return True


def validate(args, sess, epoch, dbase, enqueue_op, placeholders, tensors):
    print('\nRunning forward pass on validation set', flush=True)

    # evaluate batch size and number of batches
    batch_size = min(args.batch_size, dbase.nrof_images)
    nrof_batches = dbase.nrof_images // batch_size

    # Enqueue one epoch of image paths and labels
    files = np.expand_dims(np.array(dbase.files), 1)
    labels = np.expand_dims(np.array(dbase.labels), 1)
    control = np.ones_like(labels, np.int32) * facenet.FIXED_STANDARDIZATION * args.image.standardization

    feed_dict = placeholders.enqueue_feed_dict(files, labels, control)
    sess.run(enqueue_op, feed_dict=feed_dict)

    start_time = time.monotonic()

    outputs = {}
    for key in tensors.tensors['tensors'].keys():
        outputs[key] = []

    with tqdm(total=nrof_batches) as bar:
        for i in range(nrof_batches):
            feed_dict = placeholders.run_feed_dict(batch_size)
            output = sess.run(tensors.tensors, feed_dict=feed_dict)

            for key, value in output['tensors'].items():
                outputs[key].append(value)

            info = '[{}/{}] '.format(epoch + 1, args.train.epoch.nrof_epochs)
            bar.set_postfix_str(info + tensors.get_info_str(output))
            bar.update()

    elapsed_time = time.monotonic() - start_time
    tensors.set_elapsed_time(elapsed_time)

    for key, value in outputs.items():
        outputs[key] = np.mean(value)

    outputs = {'tensors': outputs}
    tensors.set_output(outputs)


if __name__ == '__main__':
    main()

