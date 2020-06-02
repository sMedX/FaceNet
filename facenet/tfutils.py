# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

from pathlib import Path
import tensorflow as tf


def get_model_filenames(model_dir):
    files = model_dir.glob('*')
    meta_files = [s for s in files if str(s).endswith('.meta')]

    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory {}.'.format(model_dir))

    if len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory {}.'.format(model_dir))

    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    ckpt_file = Path(ckpt.model_checkpoint_path).name
    return meta_file, ckpt_file


def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']

    # Get the list of important nodes
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('InceptionResnet') or node.name.startswith('embeddings') or
                node.name.startswith('image_batch') or node.name.startswith('phase_train')):
            whitelist_names.append(node.name)

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
                                                                              input_graph_def,
                                                                              output_node_names,
                                                                              variable_names_whitelist=whitelist_names)
    return output_graph_def


def save_freeze_graph(model_dir, output_file=None, suffix=''):
    if output_file is None:
        output_file = model_dir.joinpath(model_dir.name + suffix + '.pb')
    else:
        output_file = output_file.expanduser()

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model metagraph and checkpoint
            print('Model directory: {}'.format(model_dir))
            meta_file, ckpt_file = get_model_filenames(model_dir)

            print('Metagraph file: {}'.format(meta_file))
            print('Checkpoint file: {}'.format(ckpt_file))

            saver = tf.train.import_meta_graph(str(model_dir.joinpath(meta_file)), clear_devices=True)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
            saver.restore(sess, str(model_dir.joinpath(ckpt_file)))

            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()

            output_graph_def = freeze_graph_def(sess, input_graph_def, ['embeddings'])

            # dest_nodes = ['input:0', 'embeddings:0']
            # output_graph_def = tf.compat.v1.graph_util.extract_sub_graph(input_graph_def, dest_nodes)

        # Serialize and dump the output graph to the filesystem
        with tf.io.gfile.GFile(str(output_file), 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('{} ops in the final graph: {}'.format(len(output_graph_def.node), str(output_file)))

    return output_file


def load_model(path, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file) or
    # if it is a protobuf file with a frozen graph

    path = Path(path).expanduser()
    print('load model: {}'.format(path))

    if path.is_file():
        print('Model filename: {}'.format(path))
        with tf.io.gfile.GFile(str(path), 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        pb_file = path.joinpath(path.name + '.pb')

        if pb_file.exists():
            load_model(pb_file, input_map=input_map)
        else:
            print('Model directory: {}'.format(path))
            meta_file, ckpt_file = get_model_filenames(path)

            print('Metagraph file : {}'.format(meta_file))
            print('Checkpoint file: {}'.format(ckpt_file))

            saver = tf.train.import_meta_graph(str(path.joinpath(meta_file)), input_map=input_map)
            with tf.Session() as sess:
                saver.restore(sess, str(path.joinpath(ckpt_file)))
