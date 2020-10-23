"""Validate a face recognizer.
"""
# MIT License
#
# Copyright (c) 2020 SMedX

import click
from pathlib import Path
import tensorflow as tf

from facenet import dataset, config, facenet, tfutils


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
