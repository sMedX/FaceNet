# coding:utf-8
"""Application to print some info about model
"""
# MIT License
# Copyright (c) 2020 sMedX


import click
import importlib
from pathlib import Path
import tensorflow as tf
import numpy as np
from facenet import tfutils, config

path = Path('facenet/models/configs/inception_resnet_v1.yaml')


@click.command()
@click.option('--path', default=path, type=Path, help='Path to model.')
def main(**args_):

    with tf.Graph().as_default():
        with tf.Session() as sess:
            if str(args_['path']).endswith('yaml'):
                args = config.YAMLConfig(args_['path'])

                image_batch = tf.convert_to_tensor(np.zeros([1, 160, 160, 3]), np.float32)
                image_batch = tf.identity(image_batch, 'input')
                network = importlib.import_module(args.module)
                prelogits, end_points = network.inference(image_batch, config=args.config, phase_train=False)
                embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
            else:
                tfutils.load_model(args_['path'], input_map=None)

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

            image_placeholder = graph.get_tensor_by_name("input:0")

            feed_dict = {
                image_placeholder: np.zeros([1, 160, 160, 3]),
            }

            if tfutils.tensor_by_name_exist('phase_train:0'):
                phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
                feed_dict[phase_train_placeholder] = False

            out = sess.run(embeddings, feed_dict=feed_dict)
            print(out.shape)


if __name__ == '__main__':
    main()

