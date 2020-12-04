"""Validate a face recognizer.
"""
# MIT License
#
# Copyright (c) 2020 SMedX

import click
from pathlib import Path
import tensorflow as tf

from facenet import dataset, config, facenet, tfutils, ioutils, h5utils


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options for the application.')
def main(**options):
    options = config.Embeddings(options)

    dbase = dataset.DBase(options.dataset)
    ioutils.write_text_log(options.log_file, dbase)
    print(dbase)

    embeddings = facenet.EvaluationOfEmbeddings(dbase, options)
    ioutils.write_text_log(options.log_file, dbase)
    print(embeddings)

    if options.output.suffix == '.h5':
        h5utils.write(options.output, 'embeddings', embeddings.embeddings)
        h5utils.write(options.output, 'labels', embeddings.labels)
    else:
        with tf.io.TFRecordWriter(str(options.output)) as writer:
            for embedding, label, file in zip(embeddings.embeddings, dbase.labels, dbase.files):
                feature = {
                    'embedding': tfutils.float_feature(embedding.tolist()),
                    'label': tfutils.int64_feature(label),
                    'file': tfutils.bytes_feature(file.encode())
                    }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

    print('output file:', options.output)
    print('number of examples:', dbase.nrof_images)


if __name__ == '__main__':
    main()
