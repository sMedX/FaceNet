"""Exports trainable parameters to h5 file
"""
# MIT License
# Copyright (c) 2020 sMedX

import click
from pathlib import Path

import tensorflow.compat.v1 as tf

from facenet import tfutils, config, facenet
import facenet.models.inception_resnet_v1 as module


@click.command()
@click.option('--model_dir', default=config.default_model, type=Path,
              help='Directory with the meta graph and checkpoint files containing model parameters.')
def main(**args):
    files = config.data_dir.glob('*' + config.file_extension)

    loader = facenet.ImageLoader(config=None)

    images = [tf.expand_dims(loader(f), 0) for f in files]
    image_batch = tf.concat(images, axis=0)

    with tf.Session() as sess:
        image_batch = sess.run(image_batch)

    tfutils.export_h5(args['model_dir'], image_batch, module)


if __name__ == '__main__':
    main()
