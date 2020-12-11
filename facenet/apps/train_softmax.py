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
import importlib
from tqdm import tqdm
from pathlib import Path
import tensorflow.compat.v1 as tf
import tf_slim as slim

from facenet import nodes, ioutils, dataset, statistics, config, h5utils, facenet, tfutils


@click.command()
@click.option('--config', default=None, type=Path,
              help='Path to yaml config file with used options of the application.')
def main(**options):
    start_time = time.monotonic()
    options = config.train_softmax(__file__, options)

    # import network
    print('import model {}'.format(options.model.module))
    network = importlib.import_module(options.model.module)

    # ------------------------------------------------------------------------------------------------------------------
    dbase = dataset.DBase(options.dataset)
    ioutils.write_text_log(options.logfile, dbase)
    print('train dbase:', dbase)

    dbase_val = dataset.DBase(options.validate.dataset)
    ioutils.write_text_log(options.logfile, dbase_val)
    print('validate dbase', dbase_val)

    loader = facenet.ImageLoader(config=options.image)
    ds = {
        'train': facenet.make_train_dataset(dbase, loader, options),
        'validate': facenet.make_test_dataset(dbase_val, loader, options),
    }

    iterator = {
        'train': ds['train'].make_one_shot_iterator(),
        'validate': ds['validate'].make_initializable_iterator(),
    }

    batch = {
        'train': iterator['train'].get_next(),
        'validate': iterator['validate'].get_next(),
    }

    # ------------------------------------------------------------------------------------------------------------------
    global_step = tf.Variable(0, trainable=False, name='global_step')

    placeholders = facenet.Placeholders()

    print('Building training graph')

    image_processing = facenet.ImageProcessing(options.image)
    image_batch = image_processing(placeholders.image_batch)

    prelogits, _ = network.inference(image_batch,
                                     config=options.model.config,
                                     phase_train=placeholders.phase_train)

    logits = slim.fully_connected(prelogits, dbase.nrof_classes, activation_fn=None,
                                  weights_initializer=slim.initializers.xavier_initializer(),
                                  weights_regularizer=slim.l2_regularizer(options.model.config.weight_decay),
                                  scope='Logits', reuse=False)

    output_node_name = nodes['output']['name']
    embedding = tf.nn.l2_normalize(prelogits, 1, 1e-10, name=output_node_name)

    # Norm for the prelogits
    eps = 1e-4
    prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=options.loss.prelogits_norm_p, axis=1))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * options.loss.prelogits_norm_factor)

    # Add center loss
    prelogits_center_loss, _ = facenet.center_loss(prelogits, placeholders.label_batch, options.loss.center_alfa,
                                                   dbase.nrof_classes)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * options.loss.center_factor)

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
    learning_rate = placeholders.learning_rate
    train_op = facenet.train_op(options.train, total_loss, global_step, learning_rate, tf.global_variables())

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
                'accuracy': accuracy,
                'loss': total_loss,
                'xent': cross_entropy_mean,
                'center_loss': prelogits_center_loss,
                'prelogits_norm': prelogits_norm,
                'learning_rate': learning_rate
            }
        }

        summary = {
            'train': facenet.Summary(summary_writer, options.h5file, tag='train'),
            'validate': facenet.Summary(summary_writer, options.h5file, tag='validate')
        }

        # Training and validation loop
        for epoch in range(options.train.epoch.nrof_epochs):
            info = f'(model {options.model.path.stem}, epoch [{epoch+1}/{options.train.epoch.nrof_epochs}])'
            print('\nRunning training', info)

            # train for one epoch
            train(sess, placeholders, epoch, tensor_ops, summary['train'], batch['train'], options.train)

            # save variables and the meta graph if it doesn't exist already
            tfutils.save_variables_and_metagraph(sess, saver, options.model.path, epoch)

            # perform validation
            epoch1 = epoch + 1
            if epoch1 % options.validate.every_n_epochs == 0 or epoch1 == options.train.epoch.nrof_epochs:
                # validate(sess, placeholders,
                #          ds['validate'], iterator['validate'], batch['validate'],
                #          tensor_dict['validate'], summary['validate'], info)

                # perform face-to-face validation
                embeddings, labels = facenet.evaluate_embeddings(sess, embedding, placeholders,
                                                                 ds['validate'], iterator['validate'], batch['validate'],
                                                                 info)

                validation = statistics.FaceToFaceValidation(embeddings, labels, options.validate.validate, info=info)

                ioutils.write_text_log(options.txtfile, str(validation))
                h5utils.write_dict(options.h5file, validation.dict, group='validate')

                for key, value in validation.dict.items():
                    summary['validate'].write_tf_summary(value, tag='{}_{}'.format('validate', key))

                print(validation)

    tfutils.save_frozen_graph(model_dir=options.model.path, optimize=True)

    ioutils.write_elapsed_time(options.h5file, start_time)
    ioutils.write_elapsed_time(options.txtfile, start_time)

    print('Statistics have been saved to the h5 file: {}'.format(options.h5file))
    print('Logs have been saved to the directory: {}'.format(options.logs))
    print('Model has been saved to the directory: {}'.format(options.model.path))

    return options.model.path


def train(sess, placeholders, epoch, tensor_dict, summary, batch, options):
    start_time = time.monotonic()

    learning_rate = facenet.learning_rate_value(epoch, options.learning_rate)
    if not learning_rate:
        return False

    outputs = {key: [] for key in tensor_dict['tensor_op'].keys()}

    with tqdm(total=options.epoch.size) as bar:
        for batch_number in range(options.epoch.size):
            image_batch, label_batch = sess.run(batch)

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

