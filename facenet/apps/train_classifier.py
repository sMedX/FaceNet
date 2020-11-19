"""Validate a face recognizer.
"""
# MIT License
#
# Copyright (c) 2020 SMedX

import click
from tqdm import tqdm
from pathlib import Path

import tensorflow as tf
import numpy as np

from facenet import config, facenet, ioutils


class FaceToFaceDistanceModel:
    def __init__(self):
        self.trainable_variables = {
            'alpha': tf.Variable(initial_value=10, dtype=tf.float32, name='alpha'),
            'threshold': tf.Variable(initial_value=1, dtype=tf.float32, name='threshold'),
            'theta': tf.Variable(initial_value=0.3, dtype=tf.float32, name='theta')
        }

    def __call__(self, x, y=None):
        alpha = self.variable('alpha')
        threshold = self.variable('threshold')
        logits = tf.multiply(alpha, tf.subtract(threshold, self.distance(x, y)))
        return logits

    def __repr__(self):
        variables = {}
        for name in self.trainable_variables.keys():
            variables[name] = self.variable(name, mode='numpy')

        return (f'{self.__class__.__name__}\n'
                f'variables {variables}\n')

    def variable(self, name, mode=None):
        var = self.trainable_variables[name]
        if mode == 'numpy':
            var = tf.get_default_session().run(var)
        return var

    def distance(self, x, y):
        if y is None:
            y = x

        if isinstance(x, np.ndarray):
            theta = self.variable('theta', mode='numpy')

            y = np.transpose(y)

            norm_x = np.linalg.norm(x, axis=1, keepdims=True)
            norm_y = np.linalg.norm(y, axis=0, keepdims=True)
        else:
            theta = self.variable('theta')

            y = tf.transpose(y)

            norm_x = tf.linalg.norm(x, axis=1, keepdims=True)
            norm_y = tf.linalg.norm(y, axis=0, keepdims=True)

        length = (norm_x + norm_y) / 2
        length2 = length * length

        x1 = x / norm_x
        y1 = y / norm_y

        dx = 1 - norm_x / length  # length of (x - x1)
        dy = 1 - norm_y / length  # length of (y - y1)

        # first order of theta - (x - x1, x - x1) + (y - y1, y - y1)
        # second order of theta - (y1 - x1, x1 - x) + (y1 - x1, y - y1) + (x1 - x, y - y1)
        x1_y1 = x1 @ y1

        dist = 2 * (1 - x1_y1) + theta * theta * (dx * dx + dy * dy) + 2 * theta * (x1_y1 - x @ y / length2)

        return dist

    def predict(self, x, y=None):
        return self.distance(x, y) < self.variable('threshold', mode='numpy')


class FaceToFaceNormalizedModel:
    def __init__(self):
        self.trainable_variables = {
            'alpha': tf.Variable(initial_value=10.0, dtype=tf.float32, name='alpha'),
            'threshold': tf.Variable(initial_value=1.0, dtype=tf.float32, name='threshold')
        }

    def __call__(self, x, y=None):
        alpha = self.variable('alpha')
        threshold = self.variable('threshold')
        logits = tf.multiply(alpha, tf.subtract(threshold, self.distance(x, y)))
        return logits

    def __repr__(self):
        variables = {}
        for name in self.trainable_variables.keys():
            variables[name] = self.variable(name, mode='numpy')

        return (f'{self.__class__.__name__}\n'
                f'variables {variables}\n')

    def variable(self, name, mode=None):
        var = self.trainable_variables[name]
        if mode == 'numpy':
            var = tf.get_default_session().run(var)
        return var

    def distance(self, x, y):
        if y is None:
            y = x

        if isinstance(x, np.ndarray):
            dist = 2 * (1 - x @ np.transpose(y))
        else:
            dist = 2 * (1 - x @ tf.transpose(y))

        return dist

    def predict(self, x, y=None):
        return self.distance(x, y) < self.variable('threshold', mode='numpy')


class ConfusionMatrix:
    def __init__(self, embeddings, classifier):
        nrof_classes = len(embeddings)
        nrof_positive_class_pairs = nrof_classes
        nrof_negative_class_pairs = nrof_classes * (nrof_classes - 1) / 2

        tp = tn = fp = fn = 0

        for i in range(nrof_classes):
            for k in range(i):
                outs = classifier.predict(embeddings[i], embeddings[k])
                mean = np.mean(outs)

                fp += mean
                tn += 1 - mean

            outs = classifier.predict(embeddings[i])
            mean = np.mean(outs)

            tp += mean
            fn += 1 - mean

        tp /= nrof_positive_class_pairs
        fn /= nrof_positive_class_pairs

        fp /= nrof_negative_class_pairs
        tn /= nrof_negative_class_pairs

        self.classifier = classifier
        self.accuracy = (tp + tn) / (tp + fp + tn + fn)
        self.precision = tp / (tp + fp)
        self.tp_rate = tp / (tp + fn)
        self.tn_rate = tn / (tn + fp)

    def __repr__(self):
        return (f'{self.__class__.__name__}\n' +
                f'{str(self.classifier)}\n' +
                f'accuracy  {self.accuracy}\n' +
                f'precision {self.precision}\n' +
                f'tp rate   {self.tp_rate}\n' +
                f'tn rate   {self.tn_rate}\n')


def binary_cross_entropy_loss(logits, options):
    # define upper-triangle indices
    batch_size = options.nrof_classes_per_batch * options.nrof_examples_per_class
    triu_indices = [(i, k) for i, k in zip(*np.triu_indices(batch_size, k=1))]

    # compute labels for embeddings
    labels = []
    for i, k in triu_indices:
        if (i // options.nrof_examples_per_class) == (k // options.nrof_examples_per_class):
            # label 1 means inner class distance
            labels.append(1)
        else:
            # label 0 means across class distance
            labels.append(0)

    pos_weight = len(labels) / sum(labels) - 1

    logits = tf.gather_nd(logits, triu_indices)
    labels = tf.constant(labels, dtype=logits.dtype)

    # initialize cross entropy loss
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels, logits, pos_weight)
    loss = tf.reduce_mean(cross_entropy)

    return loss


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options for the application.')
def main(**options):
    options = config.TrainClassifier(options)

    embeddings = facenet.Embeddings(options.embeddings)
    ioutils.write_text_log(options.log_file, embeddings)

    next_elem = facenet.equal_batches_input_pipeline(embeddings.embeddings, options)

    embeddings_batch = tf.placeholder(tf.float32, shape=[None, embeddings.length], name='embeddings_batch')

    model = FaceToFaceDistanceModel()
    logits = model(embeddings_batch)
    cross_entropy = binary_cross_entropy_loss(logits, options)

    # define train operations
    global_step = tf.Variable(0, trainable=False, name='global_step')

    dtype = tf.float64
    initial_learning_rate = tf.constant(options.train.learning_rate_schedule.initial_value, dtype=dtype)
    decay_rate = tf.constant(options.train.learning_rate_schedule.decay_rate, dtype=dtype)

    if not options.train.learning_rate_schedule.decay_steps:
        decay_steps = tf.constant(options.train.epoch.size, dtype=dtype)
    else:
        decay_steps = tf.constant(options.train.learning_rate_schedule.decay_steps, dtype=dtype)

    lr_decay_factor = tf.math.pow(decay_rate, tf.math.floor(tf.cast(global_step, dtype=dtype) / decay_steps))
    learning_rate = initial_learning_rate * lr_decay_factor

    train_ops = facenet.train_op(options.train, cross_entropy, global_step, learning_rate, tf.global_variables())

    tensor_ops = {
        'global_step': global_step,
        'loss': cross_entropy,
        'vars': tf.trainable_variables(),
        'learning_rate': learning_rate
    }

    print('start training')

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for epoch in range(options.train.epoch.max_nrof_epochs):
            with tqdm(total=options.train.epoch.size) as bar:
                for _ in range(options.train.epoch.size):
                    embeddings_batch_np = session.run(next_elem)
                    feed_dict = {embeddings_batch: embeddings_batch_np}

                    _, outs = session.run([train_ops, tensor_ops], feed_dict=feed_dict)

                    postfix = f"variables {outs['vars']}, loss {outs['loss']}"
                    bar.set_postfix_str(postfix)
                    bar.update()

            info = f"epoch [{epoch + 1}/{options.train.epoch.max_nrof_epochs}], learning rate {outs['learning_rate']}"
            print(info)

            conf_mat = ConfusionMatrix(embeddings.embeddings, model)
            print(conf_mat)
            ioutils.write_text_log(options.log_file, info)
            ioutils.write_text_log(options.log_file, conf_mat)

    print(f'Model has been saved to the directory: {options.classifier.path}')


if __name__ == '__main__':
    main()
