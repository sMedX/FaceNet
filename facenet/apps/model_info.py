# coding:utf-8
"""Application to print some info about model
"""
# MIT License
# Copyright (c) 2020 sMedX


import click
from pathlib import Path
import numpy as np

import tensorflow as tf

from facenet import tfutils, config, nodes


@click.command()
@click.option('--path', default=config.default_model, type=Path, help='Path to model.')
def main(**args):

    input_node_name = '{}:0'.format(nodes['input']['name'][0])
    output_node_name = '{}:0'.format(nodes['output']['name'][0])

    with tf.Graph().as_default():
        with tf.Session() as sess:
            tfutils.load_model(args['path'], input_map=None)

            graph = tf.compat.v1.get_default_graph()

            # for i, op in enumerate(graph.get_operations()):
            #     print(i, op.name, op.type)
            #     print('\tinputs ', op.inputs)
            #     print('\toutputs', op.outputs)

            print()
            print('length of list of graph operations', len(graph.get_operations()))
            print('length of list of global variables', len(tf.compat.v1.global_variables()))

            nrof_vars = 0
            for var in tf.compat.v1.global_variables():
                nrof_vars += np.prod(var.shape)
            print('number of variables', nrof_vars)

            image_placeholder = graph.get_tensor_by_name(input_node_name)
            print('image :', image_placeholder)

            embedding = graph.get_tensor_by_name(output_node_name)
            print('output:', embedding)

            phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')

            feed_dict = {
                image_placeholder: np.zeros([1, 160, 160, 3], dtype=np.uint8),
                phase_train_placeholder: False
            }

            out = sess.run(embedding, feed_dict=feed_dict)
            print(out.shape)


if __name__ == '__main__':
    main()

