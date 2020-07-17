"""Validate a face recognizer.
"""
# MIT License
# 
# Copyright (c) 2020 SMedX

import click
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from facenet import dataset, config, statistics, facenet


# evaluation of distances
def similarity(x, y):
    return 2*(1 - x @ tf.transpose(y))


class Metrics:
    def __init__(self, embeddings, batches, threshold, alpha, far=None):
        nrof_classes = len(embeddings)
        nrof_positive_class_pairs = nrof_classes
        nrof_negative_class_pairs = nrof_classes * (nrof_classes - 1)

        tp = tn = fp = fn = 0
        positive_entropy_part = 0
        negative_entropy_part = 0

        for i, x in enumerate(embeddings):
            for k, y in enumerate(batches):
                sims = tf.cast(similarity(x, y), dtype=threshold.dtype)
                logits = tf.multiply(alpha, tf.subtract(threshold, sims))

                predict = tf.greater_equal(logits, 0)
                mean = tf.count_nonzero(predict) / tf.size(predict, out_type=tf.int64)

                if k == i:
                    logsig = tf.math.log_sigmoid(logits)
                    positive_entropy_part -= tf.reduce_mean(logsig) / nrof_positive_class_pairs

                    tp += mean / nrof_positive_class_pairs
                    fn += (1 - mean) / nrof_positive_class_pairs
                else:
                    logsig = tf.math.log_sigmoid(1 - logits)
                    negative_entropy_part -= tf.reduce_mean(logsig) / nrof_negative_class_pairs

                    fp += mean / nrof_negative_class_pairs
                    tn += (1 - mean) / nrof_negative_class_pairs

        self.cross_entropy = positive_entropy_part + negative_entropy_part

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
    embeddings = [tf.convert_to_tensor(emb) for emb in embeddings]

    # define batches
    batches = []
    for emb in embeddings:
        ds = tf.data.Dataset.from_tensor_slices(emb).shuffle(buffer_size=10, reshuffle_each_iteration=True)
        ds = ds.repeat().batch(batch_size=10)
        batches.append(tf.convert_to_tensor(ds.make_one_shot_iterator().get_next()))

    # vars and metrics
    alpha = tf.Variable(initial_value=10, dtype=tf.float64, name='alpha')
    threshold = tf.Variable(initial_value=1, dtype=tf.float64, name='threshold')

    metrics = Metrics(embeddings, batches, threshold, alpha)

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
        'variables': tf.trainable_variables(),
    }

    print('start training')

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        with tqdm(total=args.nrof_epochs) as bar:
            for _ in range(args.nrof_epochs):
                outs = session.run(ops)

                bar.set_postfix_str('variables {}, loss {}'.format(outs['variables'], outs['loss']))
                bar.update()

        metrics = Metrics(embeddings, embeddings, threshold, alpha)
        print(session.run(metrics.__dict__))


if __name__ == '__main__':
    main()
