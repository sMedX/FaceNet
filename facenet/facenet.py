"""Functions for building the face recognition network.
"""
# MIT License
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

import math
import random
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from functools import partial

from facenet import ioutils, h5utils, FaceNet


class Placeholders:
    def __init__(self, image_size):
        self.image_batch = tf.placeholder(tf.uint8, shape=[None, None, None, 3], name='image_batch')
        self.label_batch = tf.placeholder(tf.int32, shape=[None], name='label_batch')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    def train_feed_dict(self, image_batch, label_batch, learning_rate):
        return {
            self.image_batch: image_batch,
            self.label_batch: label_batch,
            self.learning_rate: learning_rate,
            self.phase_train: True,
            self.batch_size: image_batch.shape[0]
        }

    def validate_feed_dict(self, image_batch, label_batch):
        return {
            self.image_batch: image_batch,
            self.label_batch: label_batch,
            self.phase_train: False,
            self.batch_size: image_batch.shape[0]
        }

    def embedding_feed_dict(self, image_batch):
        return {
            self.image_batch: image_batch,
            self.phase_train: False,
            self.batch_size: image_batch.shape[0]
        }


def load_images(path, image_size):
    contents = tf.io.read_file(path)
    image = tf.image.decode_image(contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
    return image


def binary_cross_entropy_input_pipeline(dbase, args):
    print('Building binary cross-entropy pipeline.')

    batch_size = args.nrof_classes_per_batch * args.nrof_examples_per_class

    loader = partial(load_images, image_size=args.image.size)

    def generator():
        while True:
            files = []
            for cls in random.sample(dbase.classes, args.nrof_classes_per_batch):
                files += random.sample(cls.files, args.nrof_examples_per_class)
            yield files

    dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string)

    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

    dataset = tf.data.Dataset.zip((dataset.map(loader), dataset)).batch(batch_size)

    image_batch, label_batch = dataset.make_one_shot_iterator().get_next()

    return image_batch, label_batch


def image_processing(image_batch, args):
    image_size = tf.convert_to_tensor([args.size, args.size], name='image_size')

    image_batch = tf.identity(image_batch, 'image')
    image_batch = tf.image.resize(image_batch, size=image_size, name='resized_image')

    if args.normalization == 0:
        grayscale_image_batch = tf.image.rgb_to_grayscale(image_batch)
        min_value = tf.math.reduce_min(grayscale_image_batch)
        max_value = tf.math.reduce_max(grayscale_image_batch)
        image_batch = 2*(image_batch - min_value)/(max_value - min_value) - 1
    elif args.normalization == 1:
        image_batch = tf.image.per_image_standardization(image_batch)
    else:
        raise ValueError('Invalid image normalization algorithm')

    image_batch = tf.identity(image_batch, 'input')

    return image_batch


def make_train_dataset(dbase, map_func, args):
    data = list(zip(dbase.files, dbase.labels))
    np.random.shuffle(data)
    files, labels = map(list, zip(*data))

    images = tf.data.Dataset.from_tensor_slices(files).map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip((images, labels))
    ds = ds.shuffle(buffer_size=10*args.batch_size, reshuffle_each_iteration=True).repeat()
    ds = ds.batch(batch_size=args.batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def make_validate_dataset(ds, map_func, args, shuffle=True):
    if shuffle:
        data = list(zip(ds.files, ds.labels))
        np.random.shuffle(data)
        files, labels = map(list, zip(*data))
    else:
        files, labels = ds.files, ds.labels

    images = tf.data.Dataset.from_tensor_slices(files).map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip((images, labels)).batch(batch_size=args.batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def evaluate_embeddings(sess, embedding, placeholders, dataset, iterator, batch, info):
    print('\nEvaluation embeddings on validation set', info)

    embeddings = []
    labels = []

    nrof_batches = sess.run(tf.data.experimental.cardinality(dataset))
    sess.run(iterator.initializer)

    for _ in tqdm(range(nrof_batches)):
        image_batch, label_batch = sess.run(batch)
        embeddings.append(sess.run(embedding, feed_dict=placeholders.validate_feed_dict(image_batch)))
        labels.append(label_batch)

    return np.concatenate(embeddings), np.concatenate(labels)


def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss


def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers


def _add_loss_summaries(total_loss):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op


def train_op(args, total_loss, global_step, learning_rate, update_gradient_vars):
    print('Building train operations.')

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    optimizer = args.optimizer.lower()

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'mom':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
    
        grads = opt.compute_gradients(total_loss, update_gradient_vars)
        
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    # Add histograms for trainable variables and for gradients.
    if args.log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        op = tf.no_op(name='train')
  
    return op


def get_label_batch(label_data, batch_size, batch_index):
    nrof_examples = np.size(label_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = label_data[j:j+batch_size]
    else:
        x1 = label_data[j:nrof_examples]
        x2 = label_data[0:nrof_examples-j]
        batch = np.vstack([x1,x2])
    batch_int = batch.astype(np.int64)
    return batch_int


def get_batch(image_data, batch_size, batch_index):
    nrof_examples = np.size(image_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = image_data[j:j+batch_size,:,:,:]
    else:
        x1 = image_data[j:nrof_examples,:,:,:]
        x2 = image_data[0:nrof_examples-j,:,:,:]
        batch = np.vstack([x1,x2])
    batch_float = batch.astype(np.float32)
    return batch_float


def get_triplet_batch(triplets, batch_index, batch_size):
    ax, px, nx = triplets
    a = get_batch(ax, int(batch_size/3), batch_index)
    p = get_batch(px, int(batch_size/3), batch_index)
    n = get_batch(nx, int(batch_size/3), batch_index)
    batch = np.vstack([a, p, n])
    return batch


def learning_rate_value(epoch, config):
    if config.value is not None:
        return config.value

    if epoch >= config.schedule[-1][0]:
        return None

    for (epoch_, lr_) in config.schedule:
        if epoch < epoch_:
            return lr_


class EvaluationOfEmbeddings:
    def __init__(self, dbase, config):
        self.config = config
        self.dbase = dbase
        self.embeddings = []
        self.labels = []

        facenet = FaceNet(self.config.model)

        print('Running forward pass on images')

        map_func = partial(load_images, image_size=self.config.image.size)
        dataset = make_validate_dataset(dbase, map_func, self.config, shuffle=False)
        iterator = dataset.make_one_shot_iterator().get_next()

        with tf.Session() as sess:
            nrof_batches = sess.run(tf.data.experimental.cardinality(dataset))

            for i in tqdm(range(nrof_batches)):
                image_batch, label_batch = sess.run(iterator)

                self.embeddings.append(facenet.evaluate(image_batch))
                self.labels.append(label_batch)

        self.embeddings = np.concatenate(self.embeddings)
        self.labels = np.concatenate(self.labels)

    def __repr__(self):
        return ('{}\n'.format(self.__class__.__name__) +
                'model: {}\n'.format(self.config.model) +
                'embedding size: {}\n'.format(self.embeddings.shape))

    def write_report(self, file):
        info = 64 * '-' + '\n' + str(self)
        ioutils.write_to_file(file, info, mode='a')


class Summary:
    def __init__(self, summary_writer, h5file, tag=''):
        self._summary_writer = summary_writer
        self._h5file = h5file
        self._counts = {}
        self._tag = tag

    @staticmethod
    def get_info_str(output):
        if output.get('tensor_op'):
            output = output['tensor_op']

        info = ''
        for key, item in output.items():
            info += ', {} {:.5f}'.format(key, item)

        return info[1:]

    @property
    def tag(self):
        return self._tag

    def _count(self, name):
        if name not in self._counts.keys():
            self._counts[name] = -1
        self._counts[name] += 1
        return self._counts[name]

    def write_tf_summary(self, output, tag=None):
        if output.get('summary_op'):
            self._summary_writer.add_summary(output['summary_op'], global_step=self._count('summary_op'))

        if output.get('tensor_op'):
            output = output['tensor_op']

        if tag is None:
            tag = self.tag

        summary = tf.Summary()

        def add_summary(dct, tag):
            tag = tag + '/' if tag else ''
            for key, value in dct.items():
                if isinstance(value, dict):
                    add_summary(value, tag + key)
                else:
                    summary.value.add(tag=tag + key, simple_value=value)

        add_summary(output, tag)
        self._summary_writer.add_summary(summary, global_step=self._count(tag))

    def write_h5_summary(self, output):
        h5utils.write_dict(self._h5file, output, group=self._tag)

    def write_elapsed_time(self, value):
        tag = self._tag + '/time' if self.tag else 'time'

        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self._summary_writer.add_summary(summary, global_step=self._count('elapsed_time'))

        h5utils.write(self._h5file, tag, value)

