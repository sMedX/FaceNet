"""Validate a face recognizer.
"""
# MIT License
# 
# Copyright (c) 2020 SMedX

import click
from pathlib import Path
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from facenet import dataset, config, statistics, facenet


# evaluation of distances
def similarity(x, y):
    return 2*(1 - x @ y)


class ConfusionMatrix:
    def __init__(self, embeddings, threshold):
        nrof_classes = len(embeddings)
        nrof_positive_class_pairs = nrof_classes
        nrof_negative_class_pairs = nrof_classes * (nrof_classes - 1)/2

        tp = tn = fp = fn = 0

        for i in range(nrof_classes):
            for k in range(i+1):
                sims = similarity(embeddings[i], np.transpose(embeddings[k]))
                count = np.mean(sims < threshold)

                if i == k:
                    tp += count/nrof_positive_class_pairs
                    fn += (1 - count)/nrof_positive_class_pairs
                else:
                    fp += count/nrof_negative_class_pairs
                    tn += (1 - count)/nrof_negative_class_pairs

        self.accuracy = (tp + tn) / (tp + fp + tn + fn)
        self.precision = tp / (tp + fp)
        self.tp_rate = tp / (tp + fn)
        self.tn_rate = tn / (tn + fp)

    def __repr__(self):
        return ('\n'.format(self.__class__.__name__) +
                'accuracy  {}\n'.format(self.accuracy) +
                'precision {}\n'.format(self.precision) +
                'tp rate   {}\n'.format(self.tp_rate) +
                'tn rate   {}\n'.format(self.tn_rate))


class Metrics:
    def __init__(self, embeddings, batches, threshold, alpha, args):
        tp = tn = fp = fn = 0
        positive_part_entropy = 0
        negative_part_entropy = 0

        batch = tf.concat(batches, axis=0)
        idx1 = -tf.shape(batches[0])[0]

        for i, x in enumerate(embeddings):
            sims = similarity(x, tf.transpose(batch))
            logits = tf.multiply(alpha, tf.subtract(threshold, sims))

            size = tf.shape(batches[i])[0]
            idx1 += size
            idx2 = idx1 + size

            # cross entropy loss
            positive_logits = logits[:, idx1:idx2]
            positive_logsig = tf.math.log_sigmoid(positive_logits)
            positive_part_entropy -= tf.reduce_mean(positive_logsig)

            negative_logits = tf.concat([logits[:, :idx1], logits[:, idx2:]], axis=1)
            negative_logsig = tf.math.log_sigmoid(1 - negative_logits)
            negative_part_entropy -= tf.reduce_mean(negative_logsig)

            # confusion matrix based metrics
            predict = tf.greater_equal(logits, 0)

            positive_predict = predict[:, idx1:idx2]
            positive_mean = tf.count_nonzero(positive_predict)/tf.size(positive_predict, out_type=tf.int64)
            tp += positive_mean
            fn += 1 - positive_mean

            negative_predict = tf.concat([predict[:, :idx1], predict[:, idx2:]], axis=1)
            negative_mean = tf.count_nonzero(negative_predict)/tf.size(negative_predict, out_type=tf.int64)

            fp += negative_mean
            tn += 1 - negative_mean

        self.cross_entropy = positive_part_entropy + negative_part_entropy

        self.accuracy = (tp + tn) / (tp + fp + tn + fn)
        self.precision = tp / (tp + fp)
        self.tp_rate = tp / (tp + fn)
        self.tn_rate = tn / (tn + fp)


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options for the application.')
def main(**args_):
    args = config.Validate(args_)

    dbase = dataset.DBase(args.dataset)
    dbase.write_report(args.file)
    print(dbase)

    # define embeddings
    embeddings = facenet.EvaluationOfEmbeddings(dbase, args)
    print(embeddings)

    args = args.multi_face_validation

    embeddings = statistics.split_embeddings(embeddings.embeddings, embeddings.labels)

    # define batches
    batches = []
    for emb in embeddings:
        ds = tf.data.Dataset.from_tensor_slices(emb).shuffle(buffer_size=10, reshuffle_each_iteration=True)
        ds = ds.repeat().batch(batch_size=args.batch_size)
        batches.append(tf.convert_to_tensor(ds.make_one_shot_iterator().get_next()))

    # vars and metrics
    alpha = tf.Variable(initial_value=10, dtype=tf.float32, name='alpha')
    threshold = tf.Variable(initial_value=1, dtype=tf.float32, name='threshold')

    metrics = Metrics(embeddings, batches, threshold, alpha, args)

    # define train operations
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_ops = optimizer.minimize(metrics.cross_entropy, global_step=global_step)
    # ema = tf.train.ExponentialMovingAverage(args.moving_average_decay, global_step)
    # with tf.control_dependencies([train_ops]):
    #     train_ops = ema.apply(tf.trainable_variables())

    ops = {
        'train_ops': train_ops,
        'global_step': global_step,
        'loss': metrics.cross_entropy,
        'accuracy': metrics.accuracy,
        'variables': tf.trainable_variables(),
    }

    print('start training')

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        with tqdm(total=args.nrof_epochs) as bar:
            for _ in range(args.nrof_epochs):
                outs = session.run(ops)

                bar.set_postfix_str('variables {}, loss {} accuracy {}'.format(outs['variables'], outs['loss'], outs['accuracy']))
                bar.update()

        threshold = session.run(threshold)

        conf_mat = ConfusionMatrix(embeddings, threshold)
        print(conf_mat)


if __name__ == '__main__':
    main()
