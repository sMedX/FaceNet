# coding:utf-8
"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
# 
# Copyright (c) 2019 sMedX
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import click
import pathlib
import time
from datetime import datetime
import random
import numpy as np
import importlib
import h5py
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

import facenet
from facenet import dataset, lfw, ioutils, facenet, config

args = config.YAMLConfig(config.default_app_config(__file__))


@click.command()
@click.option('--config', default=None,
              help='Path to yaml config file with used options of the application.')
def main(**args_):
    args.update_from_file(args_['config'])

    # import network
    print('import model \'{}\''.format(args.model_def))
    network = importlib.import_module(args.model_def)
    if args.model_config is None:
        args.update_from_file(network.config_file)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    args.model_dir = pathlib.Path(args.model_dir).expanduser().joinpath(subdir)

    if args.log_dir is None:
        args.log_dir = args.model_dir.joinpath('logs')
    else:
        args.log_dir = pathlib.Path(args.log_dir).expanduser()

    stat_file_name = args.log_dir.joinpath('stat.h5')

    # Write arguments to a text file
    ioutils.write_namespace(args, args.log_dir.joinpath('arguments.yaml'))

    # store some git revision info in a text file in the log directory
    ioutils.store_revision_info(args.log_dir, sys.argv)

    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    dbase = dataset.DBase(args.data_dir, h5file=args.h5file)
    print(dbase)

    train_set, val_set = dbase.split(args.validation_set_split_ratio, args.min_nrof_val_images_per_class)
    nrof_classes = len(train_set)
    
    if args.pretrained_checkpoint is not None:
        args.pretrained_checkpoint = pathlib.Path(args.pretrained_checkpoint).expanduser()
    print('Pre-trained checkpoint: {}'.format(args.pretrained_checkpoint))

    if args.validation.dir:
        print('LFW directory: {}'.format(args.validation.dir))
        args.validation.dir = pathlib.Path(args.validation.dir).expanduser()
        # read the file containing the pairs used for testing
        pairs = lfw.read_pairs(args.validation.pairs)
        # get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(args.validation.dir, pairs)

    tf.reset_default_graph()
    tf.Graph().as_default()

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        
        # Get a list of image paths and their labels
        image_list, label_list = dataset.get_image_paths_and_labels(train_set)
        assert len(image_list) > 0, 'The training set should not be empty'
        
        val_image_list, val_label_list = dataset.get_image_paths_and_labels(val_set)

        # Create a queue that produces indices into the image_list and label_list 
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)
        
        index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch.size, 'index_dequeue')
        
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
        control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')
        
        input_queue = data_flow_ops.FIFOQueue(capacity=dbase.nrof_images,
                                    dtypes=[tf.string, tf.int32, tf.int32],
                                    shapes=[(1,), (1,), (1,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='enqueue_op')

        image_size = (args.image_size, args.image_size)
        image_batch, label_batch = facenet.create_input_pipeline(input_queue, image_size, args.nrof_preprocess_threads, batch_size_placeholder)

        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        
        print('Number of classes in training set: %d' % nrof_classes)
        print('Number of examples in training set: %d' % len(image_list))

        print('Number of classes in validation set: %d' % len(val_set))
        print('Number of examples in validation set: %d' % len(val_image_list))
        
        # Build the inference graph
        print('Building training graph')
        prelogits, _ = network.inference(image_batch,
                                         config=args.model_config,
                                         phase_train=phase_train_placeholder)

        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      weights_regularizer=slim.l2_regularizer(args.weight_decay),
                                      scope='Logits', reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Norm for the prelogits
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits)+eps, ord=args.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)

        # Add center loss
        prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch.size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        
        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer, 
            learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)
        
        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if args.pretrained_checkpoint is not None:
                print('Restoring pre-trained model: {}'.format(args.pretrained_checkpoint))
                saver.restore(sess, str(args.pretrained_checkpoint))

            # Training and validation loop
            print('Running training')
            nrof_steps = args.epoch.max_nrof_epochs*args.epoch.size
            nrof_val_samples = int(math.ceil(args.epoch.max_nrof_epochs / args.validate_every_n_epochs))   # Validate every validate_every_n_epochs as well as in the last epoch
            stat = {
                'loss': np.zeros((nrof_steps,), np.float32),
                'center_loss': np.zeros((nrof_steps,), np.float32),
                'reg_loss': np.zeros((nrof_steps,), np.float32),
                'xent_loss': np.zeros((nrof_steps,), np.float32),
                'prelogits_norm': np.zeros((nrof_steps,), np.float32),
                'accuracy': np.zeros((nrof_steps,), np.float32),
                'val_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_xent_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_accuracy': np.zeros((nrof_val_samples,), np.float32),
                'lfw_accuracy': np.zeros((args.epoch.max_nrof_epochs,), np.float32),
                'lfw_valrate': np.zeros((args.epoch.max_nrof_epochs,), np.float32),
                'learning_rate': np.zeros((args.epoch.max_nrof_epochs,), np.float32),
                'time_train': np.zeros((args.epoch.max_nrof_epochs,), np.float32),
                'time_validate': np.zeros((args.epoch.max_nrof_epochs,), np.float32),
                'time_evaluate': np.zeros((args.epoch.max_nrof_epochs,), np.float32),
                'prelogits_hist': np.zeros((args.epoch.max_nrof_epochs, 1000), np.float32),
              }

            for epoch in range(1, args.epoch.max_nrof_epochs + 1):
                step = sess.run(global_step, feed_dict=None)
                # Train for one epoch
                t = time.time()
                cont = train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, global_step, 
                    total_loss, train_op, summary_op, summary_writer, regularization_losses,
                    stat, cross_entropy_mean, accuracy,
                    prelogits, prelogits_center_loss, prelogits_norm, learning_rate)
                stat['time_train'][epoch-1] = time.time() - t
                
                if not cont:
                    break
                  
                t = time.time()
                if len(val_image_list)>0 and ((epoch-1) % args.validate_every_n_epochs == args.validate_every_n_epochs-1 or epoch==args.max_nrof_epochs):
                    validate(args, sess, epoch, val_image_list, val_label_list, enqueue_op, image_paths_placeholder, labels_placeholder, control_placeholder,
                        phase_train_placeholder, batch_size_placeholder, 
                        stat, total_loss, regularization_losses, cross_entropy_mean, accuracy, args.validate_every_n_epochs, args.image_standardization)
                stat['time_validate'][epoch-1] = time.time() - t

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, args.model_dir, subdir, epoch)

                # Evaluate on LFW
                t = time.time()
                if args.validation.dir:
                    evaluate(args, sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                             batch_size_placeholder, control_placeholder, embeddings, label_batch, lfw_paths, actual_issame,
                             step, summary_writer, stat, epoch)
                stat['time_evaluate'][epoch-1] = time.time() - t

                print('Saving statistics')
                with h5py.File(stat_file_name, 'w') as f:
                    for key, value in stat.items():
                        f.create_dataset(key, data=value)

    print('Model directory: %s' % args.model_dir)
    print('Log directory: %s' % args.log_dir)

    facenet.save_freeze_graph(model_dir=args.model_dir)

    return args.model_dir


def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step, 
      loss, train_op, summary_op, summary_writer, reg_losses,
      stat, cross_entropy_mean, accuracy, 
      prelogits, prelogits_center_loss, prelogits_norm, learning_rate):
    batch_number = 0
    
    if args.learning_rate.value > 0.0:
        lr = args.learning_rate.value
    else:
        lr = facenet.get_learning_rate_from_file(args.learning_rate.schedule_file, epoch)
        
    if lr <= 0:
        return False 

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    control_value = facenet.RANDOM_ROTATE * args.image.random_rotate + \
                    facenet.RANDOM_CROP * args.image.random_crop + \
                    facenet.RANDOM_FLIP * args.image.random_flip + \
                    facenet.FIXED_STANDARDIZATION * args.image.standardization

    control_array = np.ones_like(labels_array) * control_value
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch.size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
        tensor_list = [loss, train_op, step, reg_losses, prelogits, cross_entropy_mean, learning_rate, prelogits_norm, accuracy, prelogits_center_loss]
        if batch_number % 100 == 0:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_, summary_str = sess.run(tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_ = sess.run(tensor_list, feed_dict=feed_dict)
         
        duration = time.time() - start_time
        stat['loss'][step_-1] = loss_
        stat['center_loss'][step_-1] = center_loss_
        stat['reg_loss'][step_-1] = np.sum(reg_losses_)
        stat['xent_loss'][step_-1] = cross_entropy_mean_
        stat['prelogits_norm'][step_-1] = prelogits_norm_
        stat['learning_rate'][epoch-1] = lr_
        stat['accuracy'][step_-1] = accuracy_
        stat['prelogits_hist'][epoch-1,:] += np.histogram(np.minimum(np.abs(prelogits_), args.prelogits_hist_max), bins=1000, range=(0.0, args.prelogits_hist_max))[0]
        
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f\tCl %2.3f' %
              (epoch, batch_number+1, args.epoch.size, duration, loss_, cross_entropy_mean_, np.sum(reg_losses_), accuracy_, lr_, center_loss_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)
    return True


def validate(args, sess, epoch, image_list, label_list, enqueue_op, image_paths_placeholder, labels_placeholder, control_placeholder,
             phase_train_placeholder, batch_size_placeholder, 
             stat, loss, regularization_losses, cross_entropy_mean, accuracy, validate_every_n_epochs, image_standardization):
  
    print('Running forward pass on validation set')

    nrof_batches = len(label_list) // args.validation.batch_size
    nrof_images = nrof_batches * args.validation.batch_size
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_list[:nrof_images]),1)
    image_paths_array = np.expand_dims(np.array(image_list[:nrof_images]),1)
    control_array = np.ones_like(labels_array, np.int32)*facenet.FIXED_STANDARDIZATION * image_standardization
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})

    loss_array = np.zeros((nrof_batches,), np.float32)
    xent_array = np.zeros((nrof_batches,), np.float32)
    accuracy_array = np.zeros((nrof_batches,), np.float32)

    # Training loop
    start_time = time.time()
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: args.lfw_batch_size}
        loss_, cross_entropy_mean_, accuracy_ = sess.run([loss, cross_entropy_mean, accuracy], feed_dict=feed_dict)
        loss_array[i], xent_array[i], accuracy_array[i] = (loss_, cross_entropy_mean_, accuracy_)
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')

    duration = time.time() - start_time

    val_index = (epoch-1)//validate_every_n_epochs
    stat['val_loss'][val_index] = np.mean(loss_array)
    stat['val_xent_loss'][val_index] = np.mean(xent_array)
    stat['val_accuracy'][val_index] = np.mean(accuracy_array)

    print('Validation Epoch: %d\tTime %.3f\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f' %
          (epoch, duration, np.mean(loss_array), np.mean(xent_array), np.mean(accuracy_array)))


# evaluate(args, sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
#          batch_size_placeholder, control_placeholder, embeddings, label_batch, lfw_paths, actual_issame,
#          log_dir, step, summary_writer, stat, epoch)


def evaluate(args, sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
             batch_size_placeholder, control_placeholder, embeddings, labels, image_paths, actual_issame,
             step, summary_writer, stat, epoch):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')

    validation = args.validation
    
    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(actual_issame)*2  # nrof_pairs * nrof_images_per_pair
    nrof_flips = 2 if validation.use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    labels_array = np.expand_dims(np.arange(0, nrof_images), 1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths), nrof_flips), 1)
    control_array = np.zeros_like(labels_array, np.int32)

    if args.image.standardization:
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION

    if args.validation.use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*facenet.FLIP
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
    embedding_size = int(embeddings.get_shape()[1])
    assert nrof_images % validation.batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // validation.batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder: validation.batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
    if validation.use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:, :embedding_size] = emb_array[0::2, :]
        embeddings[:, embedding_size:] = emb_array[1::2, :]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    _, _, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame,
                                                     nrof_folds=validation.nrof_folds,
                                                     distance_metric=validation.distance_metric,
                                                     subtract_mean=validation.subtract_mean)
    
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(args.log_dir.joinpath('lfw_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))
    stat['lfw_accuracy'][epoch-1] = np.mean(accuracy)
    stat['lfw_valrate'][epoch-1] = val


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = model_dir.joinpath('model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = model_dir.joinpath('model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not metagraph_filename.exists():
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  

if __name__ == '__main__':
    main()

