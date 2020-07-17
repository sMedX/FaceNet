"""Validate a face recognizer.
"""
# MIT License
# 
# Copyright (c) 2020 SMedX

import click
from pathlib import Path
import tensorflow as tf
import numpy as np
from facenet import dataset, config, statistics, facenet


# start_time = ioutils.get_time()
# https://medium.com/@prasad.pai/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab
# https://www.tensorflow.org/tutorials/keras/classification?hl=ru

# https://habr.com/ru/post/458170/
# https://stackoverflow.com/questions/51762406/what-is-the-tensorflow-loss-equivalent-of-binary-cross-entropy
# https://www.machinelearningmastery.ru/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a/
#
# evaluation of distances
def similarity(x, y):
    return 2*(1 - x @ tf.transpose(y))


class ConfusionMatrix:
    def __init__(self, tp, tn, fp, fn):
        self.accuracy = (tp + tn) / (tp + fp + tn + fn)
        self.precision = tp / (tp + fp)
        self.tp_rate = tp / (tp + fn)
        self.tn_rate = tn / (tn + fp)


class Metrics:
    def __init__(self, embeddings, batches, threshold, alpha, far=None):
        nrof_classes = len(embeddings)
        nrof_positive_class_pairs = nrof_classes
        nrof_negative_class_pairs = nrof_classes * (nrof_classes - 1)

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        positive_entropy_part = 0
        negative_entropy_part = 0

        for i, x in enumerate(embeddings):
            for k, y in enumerate(batches):
                sims = similarity(x, y)
                logits = tf.multiply(alpha, tf.subtract(threshold, sims))

                predict = tf.greater(logits, 0)
                mean = tf.count_nonzero(predict) / tf.size(predict, out_type=tf.int64)

                if k == i:
                    logsig = tf.math.log_sigmoid(logits)

                    tp += mean / nrof_positive_class_pairs
                    fn += (1 - mean) / nrof_positive_class_pairs
                    positive_entropy_part -= tf.reduce_mean(logsig) / nrof_positive_class_pairs
                else:
                    logsig = tf.math.log_sigmoid(1 - logits)

                    fp += mean / nrof_negative_class_pairs
                    tn += (1 - mean) / nrof_negative_class_pairs
                    negative_entropy_part -= tf.reduce_mean(logsig) / nrof_negative_class_pairs

        self.cross_entropy = positive_entropy_part + negative_entropy_part
        self.conf_matrix = ConfusionMatrix(tp, tn, fp, fn).__dict__


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

    embeddings = statistics.split_embeddings(embeddings.embeddings, embeddings.labels)
    embeddings = [tf.convert_to_tensor(emb) for emb in embeddings]

    # define batches
    datasets = [tf.data.Dataset.from_tensor_slices(emb).repeat().batch(batch_size=5) for emb in embeddings]
    batches = [tf.convert_to_tensor(ds.make_one_shot_iterator().get_next()) for ds in datasets]

    # vars and metrics
    alpha = tf.Variable(initial_value=1, dtype=tf.float32, name='alpha')
    threshold = tf.Variable(initial_value=1, dtype=tf.float32, name='threshold')

    metrics = Metrics(embeddings, batches, threshold, alpha)

    # define train operations
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_ops = optimizer.minimize(metrics.cross_entropy, global_step=global_step)

    tensor_ops = {
        'global_step': global_step,
        'loss': metrics.cross_entropy,
        'alpha': alpha,
        'threshold': threshold
    }

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for i in range(1000):
            session.run(train_ops)

        outputs = session.run([tensor_ops, metrics.conf_matrix])
        for out in outputs:
            print(out)


if __name__ == '__main__':
    main()
