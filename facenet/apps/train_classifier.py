"""Validate a face recognizer.
"""
# MIT License
#
# Copyright (c) 2020 SMedX

import click
from tqdm import tqdm
from pathlib import Path
import itertools

import tensorflow as tf
import random
import numpy as np

from facenet import dataset, config, facenet, tfutils, ioutils


# evaluation of similarity
def similarity(x, y):
    return 2*(1 - x @ y)


class ConfusionMatrix:
    def __init__(self, embeddings, threshold=1):
        nrof_classes = len(embeddings)
        nrof_positive_class_pairs = nrof_classes
        nrof_negative_class_pairs = nrof_classes * (nrof_classes - 1)/2

        tp = tn = fp = fn = 0

        for i in range(nrof_classes):
            for k in range(i+1):
                sims = similarity(embeddings[i], np.transpose(embeddings[k]))
                mean = np.mean(sims < threshold)

                if i == k:
                    tp += mean
                    fn += 1 - mean
                else:
                    fp += mean
                    tn += 1 - mean

        tp /= nrof_positive_class_pairs
        fn /= nrof_positive_class_pairs

        fp /= nrof_negative_class_pairs
        tn /= nrof_negative_class_pairs

        self.threshold = threshold
        self.accuracy = (tp + tn) / (tp + fp + tn + fn)
        self.precision = tp / (tp + fp)
        self.tp_rate = tp / (tp + fn)
        self.tn_rate = tn / (tn + fp)

    def __repr__(self):
        return ('\n'.format(self.__class__.__name__) +
                'threshold {}\n'.format(self.threshold) +
                'accuracy  {}\n'.format(self.accuracy) +
                'precision {}\n'.format(self.precision) +
                'tp rate   {}\n'.format(self.tp_rate) +
                'tn rate   {}\n'.format(self.tn_rate))


def binary_cross_entropy_input_pipeline(embeddings, options):
    print('Building binary cross-entropy pipeline.')

    if not options.nrof_classes_per_batch:
        options.nrof_classes_per_batch = len(embeddings)

    batch_size = options.nrof_classes_per_batch * options.nrof_examples_per_class

    def generator():
        while True:
            embs = []
            for embeddings_per_class in random.sample(embeddings, options.nrof_classes_per_batch):
                embs += random.sample(embeddings_per_class.tolist(), options.nrof_examples_per_class)
            yield embs

    ds = tf.data.Dataset.from_generator(generator, output_types=tf.float32)
    ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

    ds = ds.batch(batch_size)
    next_elem = ds.make_one_shot_iterator().get_next()

    return next_elem


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
        if (i // options.nrof_examples_per_class) == (k // options.nrof_examples_per_class):
            # label 1 means inner class distance
            labels.append(1)
        else:
            # label 0 means across class distance
            labels.append(0)

    pos_weight = len(labels)/sum(labels) - 1

    # initialize cross entropy loss
    distances = similarity(embeddings, tf.transpose(embeddings))
    distances = tf.gather_nd(distances, triu_indices)

    logits = tf.multiply(alpha, tf.subtract(threshold, distances))
    labels = tf.constant(labels, dtype=logits.dtype)

    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels, logits, pos_weight)
    loss = tf.reduce_mean(cross_entropy)

    return loss, loss_vars


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options for the application.')
def main(**options):
    options = config.TrainClassifier(options)

    dbase = dataset.DBase(options.dataset)
    ioutils.write_text_log(options.log_file, dbase)
    print(dbase)

    embeddings = facenet.EvaluationOfEmbeddings(dbase, options)
    ioutils.write_text_log(options.log_file, embeddings)
    print(embeddings)

    embeddings = embeddings.split()
    next_elem = binary_cross_entropy_input_pipeline(embeddings, options)

    embeddings_size = embeddings[0].shape[1]
    embeddings_batch = tf.placeholder(tf.float32, shape=[None, embeddings_size], name='embeddings_batch')
    cross_entropy, loss_vars = binary_cross_entropy_loss(embeddings_batch, options)

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

            info = f"epoch [{epoch+1}/{options.train.epoch.max_nrof_epochs}], learning rate {outs['learning_rate']}"
            print(info)

            conf_mat = ConfusionMatrix(embeddings, threshold=outs['vars'][1])
            print(conf_mat)


if __name__ == '__main__':
    main()
