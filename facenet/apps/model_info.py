# coding:utf-8
"""Application to print some info about model
"""
# MIT License
# Copyright (c) 2020 sMedX


import click
from pathlib import Path

import tensorflow.compat.v1 as tf

from facenet import tfutils, config, nodes


@click.command()
@click.option('--config', default=config.default_model_path, type=Path,
              help='Path to directory with model.')
def main(**options):
    cfg = config.load_config(__file__, options)
    cfg.model.path = Path(cfg.model.path).expanduser()

    input_node_name = nodes['input']['name'] + ':0'
    output_node_name = nodes['output']['name'] + ':0'

    with tf.Graph().as_default():
        with tf.Session() as sess:
            fvars = cfg.model.path / 'variables.txt'

            tfutils.load_model(cfg.model.path)
            graph = tf.get_default_graph()

            print()
            print('length of list of graph operations', len(graph.get_operations()))
            print('length of list of global variables', len(tf.global_variables()))

            image_placeholder = graph.get_tensor_by_name(input_node_name)
            print('image :', image_placeholder)

            embedding = graph.get_tensor_by_name(output_node_name)
            print('output:', embedding)

            phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')
            print('output:', phase_train_placeholder)

            print(f'output list of trainable variables {fvars}')

            with fvars.open('w') as f:
                f.write('-----------------------------\n')
                f.write(f'number of trainable variables {len(tf.trainable_variables())}\n')
                f.write('-----------------------------\n')

                for i, var in enumerate(tf.trainable_variables()):
                    f.write(f'{i}) {var}\n')

    with tf.Graph().as_default():
        with tf.Session() as sess:
            fops = cfg.model.path / 'operations.txt'

            tfutils.load_frozen_graph(cfg.model.path)
            graph = tf.get_default_graph()

            print(f'output list of operations from frozen graph {fops}')

            with fops.open('w') as f:
                for i, op in enumerate(graph.get_operations()):
                    f.write(f'{i}) {op.name} {op.type}\n')

                    f.write(f'---  inputs [{len(op.inputs)}] {op.inputs}\n')
                    for input_tensor in op.inputs:
                        f.write(f'            {input_tensor}\n')

                    f.write(f'--- outputs [{len(op.outputs)}] {op.outputs[0]}\n')
                    for output in op.outputs[1:]:
                        f.write(f'            {output}\n')

                    f.write(f'---  values {op.values}\n')


if __name__ == '__main__':
    main()

