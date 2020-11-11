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
    alpha = tf.Variable(initial_value=1., dtype=tf.float32, name='alpha')
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
    distances = 2 * (1 - embeddings @ tf.transpose(embeddings))
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
    learning_rate = 0.01

    train_ops = facenet.train_op(options.train, cross_entropy, global_step, learning_rate, tf.global_variables())

    tensor_ops = {
        'global_step': global_step,
        'loss': cross_entropy,
        'vars': tf.trainable_variables(),
    }

    print('start training')

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        with tqdm(total=options.train.epoch.max_nrof_epochs) as bar:
            for _ in range(options.train.max_nrof_epochs):
                embeddings_batch_np = session.run(next_elem)
                feed_dict = {embeddings_batch: embeddings_batch_np}

                _, outs = session.run([train_ops, tensor_ops], feed_dict=feed_dict)

                postfix = f"variables {outs['vars']}, loss {outs['loss']}"
                bar.set_postfix_str(postfix)
                bar.update()


if __name__ == '__main__':
    main()
