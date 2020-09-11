"""Imports a model metagraph and checkpoint file, converts the variables to constants
and exports the model as a graphdef protobuf
"""
# MIT License
# Copyright (c) 2020 sMedX

import click
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

from facenet import tfutils, config, nodes


@click.command()
@click.option('--model_dir', default=config.default_model, type=Path,
              help='Directory with the meta graph and checkpoint files containing model parameters.')
@click.option('--as_text', default=0, type=int,
              help='Writes the graph as an ASCII proto.')
@click.option('--strip', default=1, type=int,
              help='Removes unused nodes from a graph file.')
@click.option('--optimize', default=1, type=int,
              help='Applies optimize_for_inference for exported graph.')
def main(**args):
    pb_file = tfutils.save_freeze_graph(args['model_dir'],
                                        strip=args['strip'],
                                        optimize=args['optimize'],
                                        as_text=args['as_text'])

    input_node_name = nodes['input']['name'] + ':0'
    output_node_name = nodes['output']['name'] + ':0'

    with tf.Graph().as_default():
        with tf.Session() as sess:
            tfutils.load_frozen_graph(args['model_dir'])
            graph = tf.get_default_graph()

            image_size = sess.run(graph.get_tensor_by_name('image_size:0'))
            print(image_size)
            height = image_size[0]
            width = image_size[1]

            image_placeholder = graph.get_tensor_by_name(input_node_name)
            print('image :', image_placeholder)

            embedding = graph.get_tensor_by_name(output_node_name)
            print('output:', embedding)

            phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')

            feed_dict = {
                image_placeholder: np.zeros([1, width, height, 3], dtype=np.uint8),
                phase_train_placeholder: False
            }

            out = sess.run(embedding, feed_dict=feed_dict)
            print(out.shape)


if __name__ == '__main__':
    main()
