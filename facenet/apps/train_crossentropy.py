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
import tensorflow as tf
from facenet import ioutils, dataset, statistics, config, h5utils, facenet, tfutils

batch_size = 8


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


# evaluation of distances
def squared_distance(x, y):
    return 2 * (1 - x @ tf.transpose(y))


# binary cross entropy
def binary_cross_entropy(embeddings):
    alpha = tf.Variable(initial_value=10., dtype=tf.float32, name='alpha')
    threshold = tf.Variable(initial_value=1., dtype=tf.float32, name='threshold')

    positive_part_entropy = 0.
    negative_part_entropy = 0.

    for i in range(batch_size):
        idx1 = i * batch_size
        idx2 = (i + 1) * batch_size
        embs = embeddings[idx1:idx2, :]

        features = squared_distance(embeddings, embs)
        logits = tf.multiply(alpha, tf.subtract(threshold, features))
        probability = tf.math.sigmoid(logits)

        positive_probability = probability[idx1:idx2, :]
        positive_part_entropy -= tf.reduce_mean(tf.math.log(positive_probability))

        negative_probability = 1 - tf.concat([probability[:idx1, :], probability[idx2:, :]], axis=0)
        negative_part_entropy -= tf.reduce_mean(tf.math.log(negative_probability))

    loss = (positive_part_entropy + negative_part_entropy) / batch_size

    return loss


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options of the application.')
def main(**args_):
    start_time = time.monotonic()
    args = config.TrainOptions(args_, subdir=config.subdir())

    # import network
    print('import model {}'.format(args.model.module))
    network = importlib.import_module(args.model.module)

    # ------------------------------------------------------------------------------------------------------------------
    dbase = dataset.DBase(args.dataset)
    ioutils.write_text_log(args.txtfile, str(dbase))
    print('train dbase:', dbase)

    dbase_val = dataset.DBase(args.validate.dataset)
    ioutils.write_text_log(args.txtfile, str(dbase_val))
    print('validate dbase', dbase_val)

    map_func = partial(facenet.load_images, image_size=args.image.size)

    val_ds = facenet.make_validate_dataset(dbase_val, map_func, args)
    val_iter = val_ds.make_initializable_iterator()
    val_elem = val_iter.get_next()

    batches = []
    for cls in tqdm(dbase.classes):
        files = cls.files
        ds = tf.data.Dataset.from_tensor_slices(files).map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.shuffle(buffer_size=100, reshuffle_each_iteration=True)
        ds = ds.repeat().batch(batch_size=batch_size, drop_remainder=True)
        ds = ds.make_one_shot_iterator().get_next()
        batches.append(ds)

    # ------------------------------------------------------------------------------------------------------------------
    print('Building training graph')

    global_step = tf.Variable(0, trainable=False, name='global_step')

    placeholders = Placeholders()

    image_batch = facenet.image_processing(placeholders.image_batch, args.image)

    prelogits, _ = network.inference(image_batch, config=args.model.config, phase_train=placeholders.phase_train)
    embedding = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embedding')
    loss = binary_cross_entropy(embedding)

    learning_rate = tf.train.exponential_decay(placeholders.learning_rate, global_step,
                                               args.train.learning_rate.decay_epochs * args.train.epoch.size,
                                               args.train.learning_rate.decay_factor, staircase=True)

    train_op = facenet.train_op(args.train, loss, global_step, learning_rate, tf.global_variables())

    # Create a saver
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Start running operations on the Graph.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        summary_writer = tf.summary.FileWriter(args.logs, sess.graph)

        tfutils.restore_checkpoint(saver, sess, args.model.checkpoint)
        tf.train.global_step(sess, global_step)
        sess.run(global_step.initializer)

        tensor_ops = {
            'train_op': train_op,
            'summary_op': summary_op,
            'tensor_op': {
                'loss': loss,
                'learning_rate': learning_rate
            }
        }

        summary = {
            'train': facenet.Summary(summary_writer, args.h5file, tag='train'),
            'validate': facenet.Summary(summary_writer, args.h5file, tag='validate')
        }

        # Training and validation loop
        for epoch in range(args.train.epoch.nrof_epochs):
            info = '(model {}, epoch [{}/{}])'.format(args.model.path.stem, epoch+1, args.train.epoch.nrof_epochs)

            # train for one epoch
            train(args, sess, placeholders, epoch, tensor_ops, summary['train'], batches, info)

            # save variables and the meta graph if it doesn't exist already
            tfutils.save_variables_and_metagraph(sess, saver, args.model.path, epoch)

            # perform validation
            epoch1 = epoch + 1
            if epoch1 % args.validate.every_n_epochs == 0 or epoch1 == args.train.epoch.nrof_epochs:
                embeddings, labels = facenet.evaluate_embeddings(sess, embedding, placeholders,
                                                                 val_ds, val_iter, val_elem, info)

                validation = statistics.FaceToFaceValidation(embeddings, labels, args.validate.validate, info=info)

                ioutils.write_text_log(args.txtfile, str(validation))
                h5utils.write_dict(args.h5file, validation.dict, group='validate')

                for key, value in validation.dict.items():
                    summary['validate'].write_tf_summary(value, tag='{}_{}'.format('validate', key))

                print(validation)

    tfutils.save_freeze_graph(model_dir=args.model.path, optimize=True)

    ioutils.write_elapsed_time(args.h5file, start_time)
    ioutils.write_elapsed_time(args.txtfile, start_time)

    print('Statistics have been saved to the h5 file: {}'.format(args.h5file))
    print('Logs have been saved to the directory: {}'.format(args.logs))
    print('Model has been saved to the directory: {}'.format(args.model.path))

    return args.model.path


def train(args, sess, placeholders, epoch, tensor_dict, summary, batches, info):
    print('\nRunning training', info)
    start_time = time.monotonic()

    learning_rate = facenet.learning_rate_value(epoch, args.train.learning_rate)
    if not learning_rate:
        return False

    outputs = {key: [] for key in tensor_dict['tensor_op'].keys()}

    epoch_size = args.train.epoch.size
    import numpy as np

    with tqdm(total=epoch_size) as bar:
        for batch_number in range(epoch_size):
            image_batch = np.random.choice(batches, size=batch_size, replace=False).tolist()
            image_batch = sess.run(tf.concat(image_batch, axis=0))

            feed_dict = placeholders.train_feed_dict(image_batch, learning_rate)
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

