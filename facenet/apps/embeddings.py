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
@click.option('--config', default=None, type=Path,
              help='Path to yaml config file with used options for the application.')
def main(**options):
    options = config.embeddings(__file__, options)

    dbase = dataset.DBase(options.dataset)
    ioutils.write_text_log(options.logfile, dbase)
    print(dbase)

    embeddings = facenet.EvaluationOfEmbeddings(dbase, options)
    ioutils.write_text_log(options.logfile, dbase)
    print(embeddings)

    if options.outfile.suffix == '.h5':
        h5utils.write(options.outfile, 'embeddings', embeddings.embeddings)
        h5utils.write(options.outfile, 'labels', embeddings.labels)
    else:
        with tf.io.TFRecordWriter(str(options.outfile)) as writer:
            for embedding, label, file in zip(embeddings.embeddings, dbase.labels, dbase.files):
                feature = {
                    'embedding': tfutils.float_feature(embedding.tolist()),
                    'label': tfutils.int64_feature(label),
                    'file': tfutils.bytes_feature(file.encode())
                    }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

    print('output file:', options.outfile)
    print('number of examples:', dbase.nrof_images)


if __name__ == '__main__':
    main()
