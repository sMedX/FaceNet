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

from tqdm import tqdm
from loguru import logger
from pathlib import Path

import random
import numpy as np
import tensorflow as tf


from facenet import nodes, h5utils, FaceNet


def inputs(config):
    return tf.keras.Input([config.size, config.size, 3])


def softmax_cross_entropy_with_logits(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


class ImageLoader:
    def __init__(self, config=None):
        self.height = config.size
        self.width = config.size

    def __call__(self, path):
        contents = tf.io.read_file(path)
        image = tf.image.decode_image(contents, channels=3)
        image = tf.image.resize_with_crop_or_pad(image, self.height, self.width)
        return image


class ImageProcessing(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()

        self.input_node_name = nodes['input']['name']

        self.config = config
        self.image_size = tf.constant([self.config.size, self.config.size], name='image_size')
        self.eps = 1e-3

    def call(self, image_batch, **kwargs):
        image_batch = tf.identity(image_batch, name=self.input_node_name)
        image_batch = tf.cast(image_batch, dtype=tf.float32, name='float_image')
        image_batch = tf.image.resize(image_batch, size=self.image_size, name='resized_image')

        if self.config.normalization == 0:
            min_value = tf.math.reduce_min(image_batch, axis=[-1, -2, -3], keepdims=True)
            max_value = tf.math.reduce_max(image_batch, axis=[-1, -2, -3], keepdims=True)
            dynamic_range = tf.math.maximum(max_value - min_value, self.eps)

            image_batch = (2 * image_batch - (min_value + max_value)) / dynamic_range

        elif self.config.normalization == 1:
            image_batch = tf.image.per_image_standardization(image_batch)
        else:
            raise ValueError('Invalid image normalization algorithm')

        image_batch = tf.identity(image_batch, name=self.__class__.__name__ + '_output')

        return image_batch


def equal_batches_input_pipeline(embeddings, config):
    """
    Building equal batches input pipeline, for example, used in binary cross-entropy pipeline.

    :param embeddings: 
    :param config: 
    :return: 
    """""
    if not config.nrof_classes_per_batch:
        config.nrof_classes_per_batch = len(embeddings)

    if not config.nrof_examples_per_class:
        config.nrof_examples_per_class = round(0.1*sum([len(embs) for embs in embeddings]) / len(embeddings))
        config.nrof_examples_per_class = max(config.nrof_examples_per_class, 1)

    print('building equal batches input pipeline.')
    print('number of classes per batch ', config.nrof_classes_per_batch)
    print('number of examples per class', config.nrof_examples_per_class)

    def generator():
        while True:
            embs = []
            for embeddings_per_class in random.sample(embeddings, config.nrof_classes_per_batch):
                embs += random.sample(embeddings_per_class.tolist(), config.nrof_examples_per_class)
            yield embs

    ds = tf.data.Dataset.from_generator(generator, output_types=tf.float32)
    ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

    batch_size = config.nrof_classes_per_batch * config.nrof_examples_per_class
    ds = ds.batch(batch_size)

    next_elem = ds.make_one_shot_iterator().get_next()

    return next_elem


def dataset(files, labels, loader, batch_size, repeat=False, buffer_size=None):
    """

    :param files:
    :param labels:
    :param loader:
    :param batch_size:
    :param repeat:
    :param buffer_size:
    :return:
    """

    data = list(zip(files, labels))
    np.random.shuffle(data)
    files, labels = map(list, zip(*data))

    images = tf.data.Dataset.from_tensor_slices(files).map(loader, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip((images, labels))

    if buffer_size is not None:
        buffer_size *= batch_size
        ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    info = (f'{ds}\n' +
            f'batch size: {batch_size}\n' +
            f'buffer size: {buffer_size}\n' +
            f'repeat: {repeat}\n' +
            f'cardinality: {ds.cardinality()}')

    logger.info('\n' + info)

    return ds


# def make_test_dataset(dbase, loader, config):
#     files, labels = dbase.files, dbase.labels
#
#     images = tf.data.Dataset.from_tensor_slices(files).map(loader)
#     labels = tf.data.Dataset.from_tensor_slices(labels)
#     ds = tf.data.Dataset.zip((images, labels))
#
#     ds = ds.batch(batch_size=config.batch_size)
#     ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
#
#     logger.info(f'batch_size: {config.batch_size}')
#     logger.info(f'cardinality: {ds.cardinality()}')
#
#     return ds


def evaluate_embeddings(model, dset):
    """
    Evaluate embeddings for given data set
    :param model:
    :param dset:
    :return:
    """

    embeddings_ = []
    labels_ = []

    for images, labels in tqdm(dset):
        embeddings = model(images)

        embeddings_.append(embeddings)
        labels_.append(labels)

    return np.concatenate(embeddings_), np.concatenate(labels_)


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


def split_embeddings(embeddings, labels):
    list_of_embeddings = []
    for label in np.unique(labels):
        emb_array = embeddings[label == labels]
        list_of_embeddings.append(emb_array)
    return list_of_embeddings


class Embeddings:
    def __init__(self, config):
        self.config = config
        self.file = Path(config.path).expanduser()

        embeddings = h5utils.read(self.file, 'embeddings')
        labels = h5utils.read(self.file, 'labels')

        self.embeddings = split_embeddings(embeddings, labels)

        if self.config.nrof_classes:
            if self.nrof_classes > self.config.nrof_classes:
                labels = [_ for _ in range(self.nrof_classes)]
                labels = random.sample(labels, self.config.nrof_classes)

                self.embeddings = [self.embeddings[label] for label in labels]

        if self.config.max_nrof_images:
            for idx, emb in enumerate(self.embeddings):
                nrof_images = emb.shape[0]

                if nrof_images > self.config.max_nrof_images:
                    labels = [_ for _ in range(nrof_images)]
                    labels = random.sample(labels, self.config.max_nrof_images)

                    self.embeddings[idx] = self.embeddings[idx][labels, :]

    def __repr__(self):
        """Representation of the embeddings"""
        data = [len(e) for e in self.embeddings]

        embeddings = np.concatenate(self.embeddings, axis=0)
        norm = np.linalg.norm(embeddings, axis=1)

        info = (f'{self.__class__.__name__}\n' +
                f'Input file {self.file}\n' +
                f'Number of classes {self.nrof_classes} \n' +
                f'Number of images {self.nrof_images}\n' +
                f'Minimal number of images in class {min(data)}\n' +
                f'Maximal number of images in class {max(data)}\n' +
                '\n' +
                f'Minimal embedding {np.min(norm)}\n' +
                f'Maximal embedding {np.max(norm)}\n' +
                f'Mean embedding {np.mean(norm)}\n'
                )

        return info

    @property
    def nrof_classes(self):
        return len(self.embeddings)

    @property
    def nrof_images(self):
        return sum([len(e) for e in self.embeddings])

    @property
    def length(self):
        return self.embeddings[0].shape[1]

    def data(self, normalize=False):

        embeddings = self.embeddings

        if normalize:
            for idx in range(self.nrof_classes):
                embeddings[idx] /= np.linalg.norm(self.embeddings[idx], axis=1, keepdims=True)

        return embeddings


class EvaluationOfEmbeddings:
    def __init__(self, dbase, config):
        self.config = config
        self.dbase = dbase
        self.embeddings = []
        self.labels = []

        facenet = FaceNet(self.config.model)

        print('Running forward pass on images')
        loader = ImageLoader(config=config.image)
        dataset = make_test_dataset(dbase, loader, self.config)
        iterator = dataset.make_one_shot_iterator().get_next()

        with tf.Session() as sess:
            nrof_batches = sess.run(tf.data.experimental.cardinality(dataset))

            for _ in tqdm(range(nrof_batches)):
                image_batch, label_batch = sess.run(iterator)

                embeddings = facenet.evaluate(image_batch)

                self.embeddings.append(embeddings)
                self.labels.append(label_batch)

        self.embeddings = np.concatenate(self.embeddings)
        self.labels = np.concatenate(self.labels)

    def __repr__(self):
        return ('{}\n'.format(self.__class__.__name__) +
                'model: {}\n'.format(self.config.model) +
                'embedding size: {}\n'.format(self.embeddings.shape))

    def split(self):
        list_of_embeddings = []

        for label in np.unique(self.labels):
            emb_array = self.embeddings[label == self.labels]
            list_of_embeddings.append(emb_array)
        return list_of_embeddings


class ConstantLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
        Learning rate schedule that return constant value
    """
    def __init__(self, value, name=None):
        super().__init__()
        self.value = tf.convert_to_tensor(value, dtype=tf.float32)
        self.name = name

    def __call__(self, step):
        return self.value

    def get_config(self):
        return self.__dict__


class PiecewiseConstantLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
        Learning rate schedule for piecewise-constant learning rate
    """
    def __init__(self, epochs, size, values, name=None):
        super().__init__()
        self.boundaries = [epoch*size for epoch in epochs]
        self.size = size
        self.values = values
        self.name = name

    def __call__(self, step):
        value = self.values[-1]

        for boundary, value in zip(self.boundaries, self.values):
            if step < boundary:
                break

        return tf.convert_to_tensor(value, dtype=tf.float32)

    def get_config(self):
        return self.__dict__


class LearningRateScheduler:
    def __init__(self, config):
        self.config = config

        if self.config.value:
            self.default_value = self.config.value
        else:
            self.default_value = None

    def __call__(self, epoch):
        if not self.default_value:
            learning_rate = self.default_value
        else:
            learning_rate = self.config.schedule[-1][1]

            for (epoch_, learning_rate) in self.config.schedule:
                if epoch < epoch_:
                    break

        info = f'Epoch {epoch}: LearningRateScheduler reducing learning rate to {learning_rate}'
        logger.info(info)

        return learning_rate
