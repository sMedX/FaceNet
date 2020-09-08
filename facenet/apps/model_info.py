# coding:utf-8
"""Application to print some info about model
"""
# MIT License
# Copyright (c) 2020 sMedX


import click
from pathlib import Path
import numpy as np

import tensorflow.compat.v1 as tf

from facenet import tfutils, config, nodes


@click.command()
@click.option('--path', default=config.default_model, type=Path, help='Path to model.')
def main(**options):

    input_node_name = nodes['input']['name'] + ':0'
    output_node_name = nodes['output']['name'] + ':0'

    with tf.Graph().as_default():
        with tf.Session() as sess:
            tfutils.load_model(options['path'], input_map=None)

            graph = tf.get_default_graph()

            fname = options['path'].joinpath('operations.txt')
            with open(fname, 'w') as f:
                for i, op in enumerate(graph.get_operations()):
                    f.write(f'{i}) {op.name} {op.type}\n')
                    f.write(f'---  inputs {op.inputs}\n')
                    f.write(f'--- outputs {op.outputs}\n')

            fname = options['path'].joinpath('variables.txt')
            with open(fname, 'w') as f:
                for i, var in enumerate(tf.trainable_variables()):
                    f.write(f'{i}) {var}\n')

            print()
            print('length of list of graph operations', len(graph.get_operations()))
            print('length of list of global variables', len(tf.global_variables()))

            image_placeholder = graph.get_tensor_by_name(input_node_name)
            print('image :', image_placeholder)

            embedding = graph.get_tensor_by_name(output_node_name)
            print('output:', embedding)

            phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')
            batch_size_placeholder = graph.get_tensor_by_name('batch_size:0')

            feed_dict = {
                image_placeholder: np.zeros([1, 160, 160, 3], dtype=np.uint8),
                phase_train_placeholder: False,
                batch_size_placeholder: 1
            }

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            out = sess.run(embedding, feed_dict=feed_dict)
            print(out.shape)

    # tfutils.load_model(options['path'], input_map=None)
    # from tensorflow.python.platform import gfile
    # from tensorflow.python.framework import tensor_util
    # from facenet import ioutils
    #
    # pbfile = ioutils.glob_single_file(options['path'], '*.pb')
    #
    # with tf.Session() as sess:
    #     print('load graph')
    #     with gfile.FastGFile(str(pbfile), 'rb') as f:
    #         graph_def = tf.GraphDef()
    #         graph_def.ParseFromString(f.read())
    #         sess.graph.as_default()
    #         tf.import_graph_def(graph_def, name='')
    #         graph_nodes = [n for n in graph_def.node]
    #         weights = [n for n in graph_nodes if n.op == 'Const']
    #
    #         for n in weights:
    #             if n.name.startswith('InceptionResnetV1/Conv2d_1a_3x3'):
    #                 w = tensor_util.MakeNdarray(n.attr['value'].tensor)
    #                 print(n.name, w.shape)
    #                 print(w)

if __name__ == '__main__':
    main()

