# coding:utf-8
"""Application to print some info about model
"""
# MIT License
# Copyright (c) 2020 sMedX


import click
from pathlib import Path
import tensorflow as tf
import numpy as np
from facenet import facenet, config


@click.command()
@click.option('--path', default=config.default_model, type=Path,
              help='Path to model.')
def main(**args_):

    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(args_['path'], input_map=None)
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

            input = graph.get_tensor_by_name("input:0")
            print('input :', input)

            embeddings = graph.get_tensor_by_name("embeddings:0")
            print('output:', embeddings)


if __name__ == '__main__':
    main()

