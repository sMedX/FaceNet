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
def main(**args_):
    args = config.Embeddings(args_)

    dbase = dataset.DBase(args.dataset)
    dbase.write_report(args.file)
    print(dbase)

    embeddings = facenet.EvaluationOfEmbeddings(dbase, args)
    embeddings.write_report(args.file)
    print(embeddings)

    with tf.io.TFRecordWriter(str(args.tfrecord)) as writer:
        for embedding, label, file in zip(embeddings.embeddings, dbase.labels, dbase.files):
            example = tfutils.dict_to_example({
                'embedding': embedding,
                'label': label,
                'file': file
                })
            writer.write(example.SerializeToString())

    print('file of TFRecords: {}'.format(args.tfrecord))
    print('number of tf examples: {}'.format(dbase.nrof_images))


if __name__ == '__main__':
    main()
