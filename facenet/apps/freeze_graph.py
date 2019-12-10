"""Imports a model metagraph and checkpoint file, converts the variables to constants
and exports the model as a graphdef protobuf
"""
# MIT License
# 
# Copyright (c) 2019 sMedX

import os
import click
import pathlib
import tensorflow as tf
from tensorflow.python.framework import graph_util
from facenet import facenet


@click.command()
@click.option('--model_dir', type=pathlib.Path,
              help='Directory with the meta graph and checkpoint files containing model parameters')
@click.option('--output_file', default=None, type=pathlib.Path,
              help='Filename for the exported protobuf file (.pb)')
def main(**args):
    model_dir = args['model_dir'].expanduser()

    if args['output_file'] is None:
        output_file = model_dir.joinpath(model_dir.name + '.pb')
    else:
        output_file = args['output_file'].expanduser()

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model metagraph and checkpoint
            print('Model directory: {}'.format(model_dir))
            meta_file, ckpt_file = facenet.get_model_filenames(model_dir)
            
            print('Metagraph file: {}'.format(meta_file))
            print('Checkpoint file: {}'.format(ckpt_file))

            model_dir_exp = args['model_dir'].expanduser()
            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file), clear_devices=True)
            tf.get_default_session().run(tf.global_variables_initializer())
            tf.get_default_session().run(tf.local_variables_initializer())
            saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))
            
            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()
            
            # Freeze the graph def
            output_graph_def = freeze_graph_def(sess, input_graph_def, 'embeddings,label_batch')

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(str(output_file), 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('{} ops in the final graph: {}'.format(len(output_graph_def.node), str(output_file)))


def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    
    # Get the list of important nodes
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('InceptionResnet') or node.name.startswith('embeddings') or 
                node.name.startswith('image_batch') or node.name.startswith('label_batch') or
                node.name.startswith('phase_train') or node.name.startswith('Logits')):
            whitelist_names.append(node.name)

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=whitelist_names)
    return output_graph_def


if __name__ == '__main__':
    main()
