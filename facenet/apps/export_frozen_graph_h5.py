"""Exports trainable parameters to h5 file
"""
# MIT License
# Copyright (c) 2020 sMedX

import click
from pathlib import Path

import tensorflow as tf

from facenet import tfutils, config, facenet
import facenet.models.inception_resnet_v1 as module


@click.command()
@click.option('--model_dir', default=config.default_model, type=Path,
              help='Directory with the meta graph and checkpoint files containing model parameters.')
def main(**args):
    files = config.data_dir.glob('*' + config.file_extension)

    images = [tf.expand_dims(facenet.load_images(f), 0) for f in files]
    image_batch = tf.concat(images, axis=0)
    with tf.compat.v1.Session() as sess:
        image_batch = sess.run(image_batch)

        tfutils.load_frozen_graph(args['model_dir'])
        graph = tf.get_default_graph()

        image_batch = sess.run(image_batch)

        for i, op in enumerate(graph.get_nodes()):
            pass



if __name__ == '__main__':
    main()
