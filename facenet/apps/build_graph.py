# coding:utf-8
"""Building and writing graph
"""
# MIT License
# Copyright (c) 2019 sMedX


import click
import pathlib
import numpy as np
import importlib
import tensorflow as tf
from facenet import config


@click.command()
@click.option('--config', default='facenet/models/configs/inception_resnet_v1.yaml',
              help='Path to yaml config file with used options of the application.')
@click.option('--logs', default='output', help='Path to the directory to write logs.')
def main(**args_):
    args = config.YAMLConfig(args_['config'])

    # import network
    print('import model \'{}\''.format(args.model_def))
    network = importlib.import_module(args.model_def)
    if args.model_config is None:
        args.update_from_file(network.config_file)

    with tf.Graph().as_default():
        image_batch = tf.convert_to_tensor(np.zeros([1, 160, 160, 3]), np.float32)
        image_batch = tf.identity(image_batch, 'input')

        # Build the inference graph
        print('Building training graph')
        prelogits, _ = network.inference(image_batch, args.keep_probability,
                                         phase_train=True,
                                         weight_decay=args.weight_decay,
                                         config=args.model_config)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # start running operations on the graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        with sess.as_default():
            print('Writing graph to the log dir', pathlib.Path(args_['logs']).expanduser().absolute())
            writer = tf.summary.FileWriter(args_['logs'], sess.graph)
            sess.run(embeddings)
            writer.close()


if __name__ == '__main__':
    main()

