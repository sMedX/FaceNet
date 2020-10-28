"""Validate a face recognizer.
"""
# MIT License
#
# Copyright (c) 2020 SMedX

import click
from pathlib import Path
import tensorflow as tf

from facenet import dataset, config, facenet, tfutils
import numpy as np


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options for the application.')
def main(**options):
    options = config.Embeddings(options)

    dbase = dataset.DBase(options.dataset)
    dbase.write_report(options.file)
    print(dbase)

    embeddings = facenet.EvaluationOfEmbeddings(dbase, options)
    embeddings.write_report(options.file)
    print(embeddings)

    def stats(x):
        norm = np.linalg.norm(x, axis=1)
        mean_norm = np.mean(norm)
        # print(np.min(norm), np.mean(norm), np.max(norm))
        return np.min(norm)/mean_norm - 1, np.max(norm)/mean_norm - 1

    stats(embeddings.embeddings)
    embs = facenet.split_embeddings(embeddings.embeddings, embeddings.labels)
    min_ = []
    max_ = []

    for idx, emb in enumerate(embs):
        x, y = stats(emb)
        min_.append(x)
        max_.append(y)
        print(f'{idx}) ', x, y)

    print()
    print(min(min_), max(max_))
    exit(0)

    with tf.io.TFRecordWriter(str(options.tfrecord)) as writer:
        for embedding, label, file in zip(embeddings.embeddings, dbase.labels, dbase.files):
            feature = {
                'embedding': tfutils.float_feature(embedding.tolist()),
                'label': tfutils.int64_feature(label),
                # 'file': tfutils.bytes_feature(file.encode())
                }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    print('file of TFRecords: {}'.format(options.tfrecord))
    print('number of tf examples: {}'.format(dbase.nrof_images))


if __name__ == '__main__':
    main()
