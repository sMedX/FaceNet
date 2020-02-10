"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
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
from pathlib import Path
import time
import numpy as np
import importlib
import itertools
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

from facenet import dataset, lfw, ioutils, statistics, config, facenet

subdir = config.subdir()


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options of the application.')
@click.option('--learning_rate', default=None, type=float,
              help='Learning rate value')
def main(**args_):
    args = config.YAMLConfig(args_['config'])

    # import network
    print('import model \'{}\''.format(args.model.module))
    network = importlib.import_module(args.model.module)
    if args.model.config is None:
        args.model.update_from_file(network.config_file)

    args.model.path = Path(args.model.path).expanduser().joinpath(subdir)
    args.model.log_dir = args.model.path.joinpath('logs')

    # write arguments to a text file
    ioutils.write_arguments(args, args.model.log_dir.joinpath('arguments.yaml'))

    # store some git revision info in a text file in the log directory
    ioutils.store_revision_info(args.model.log_dir, sys.argv)

    np.random.seed(seed=args.seed)

    train_set = dataset.DBase(args.dataset)
    print(train_set)

    # print('Model directory: %s' % model_dir)
    # print('Log directory: %s' % log_dir)
    # if args_.pretrained_model:
    #     print('Pre-trained model: %s' % os.path.expanduser(args_.pretrained_model))

    # if args.lfw_dir:
    #     print('LFW directory: %s' % args.lfw_dir)
    #     # Read the file containing the pairs used for testing
    #     pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
    #     # Get the paths for the corresponding images
    #     lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=train_set.nrof_images,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(3,), (3,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

                if args.image.random_crop:
                    image = tf.image.random_crop(image, [args.image.size, args.image.size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image.size, args.image.size)

                if args.image.random_flip:
                    image = tf.image.random_flip_left_right(image)

                if args.image.standardization:
                    image = (tf.cast(image, tf.float32) - 127.5) / 128.0
                else:
                    image = tf.image.per_image_standardization(image)

                images.append(image)
            images_and_labels.append([images, label])

        image_batch, label_batch = tf.train.batch_join(images_and_labels, batch_size=batch_size_placeholder,
                                                       shapes=[(args.image.size, args.image.size, 3), ()],
                                                       enqueue_many=True,
                                                       capacity=4 * nrof_preprocess_threads * args.batch_size,
                                                       allow_smaller_final_batch=True)

        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, phase_train=phase_train_placeholder)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, args.model.config.embedding_size]), 3, 1)
        triplet_loss = facenet.triplet_loss(anchor, positive, negative, args.alpha)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate.decay_epochs * args.epoch.size,
                                                   args.learning_rate.decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer,
                                 learning_rate, args.moving_average_decay, tf.global_variables())

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: True})

        summary_writer = tf.summary.FileWriter(args.model.log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            facenet.restore_checkpoint(saver, sess, args.model.checkpoint)

            # Training and validation loop
            for epoch in range(args.epoch.max_nrof_epochs):
                # Train for one epoch
                cont = train(args, sess, train_set, epoch, image_paths_placeholder, labels_placeholder, label_batch,
                             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op,
                             input_queue, global_step,
                             embeddings, total_loss, train_op, summary_op, summary_writer,
                             anchor, positive, negative, triplet_loss)

                if not cont:
                    break

                # Save variables and the metagraph if it doesn't exist already
                facenet.save_variables_and_metagraph(sess, saver, summary_writer, args.model.path, subdir, epoch)

                # Evaluate on LFW
                # if args.lfw_dir:
                #     evaluate(sess, lfw_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
                #              batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op,
                #              actual_issame, args.batch_size,
                #              args.lfw_nrof_folds, log_dir, step, summary_writer, args.embedding_size)

    facenet.save_freeze_graph(model_dir=args.model.path)

    # perform validation
    if args.validation is not None:
        config_ = args.validation
        dbase = dataset.DBase(config_.dataset)
        print(dbase)

        emb = facenet.Embeddings(dbase, config_, model=args.model.path)
        emb.evaluate()
        print(emb)

        stats = statistics.Validation(emb.embeddings, dbase.labels, config_.validation)
        stats.evaluate()
        stats.write_report(path=args.model.path, dbase_info=dbase.__repr__(), emb_info=emb.__repr__())
        print(stats)

    print('Model has been saved to the directory: {}'.format(args.model.path))

    return args.model.path


def train(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue,
          global_step,
          embeddings, loss, train_op, summary_op, summary_writer,
          anchor, positive, negative, triplet_loss):

    batch_number = 0
    embedding_size = int(embeddings.get_shape()[1])

    lr = facenet.learning_rate_value(epoch, args.learning_rate)
    if lr is None:
        return False

    for batch_number in range(args.epoch.size):
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people(dataset, args.people_per_batch, args.images_per_person)

        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()

        nrof_examples = args.people_per_batch * args.images_per_person
        labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

        emb_array = np.zeros((nrof_examples, embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))

        for i in range(nrof_batches):
            batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                       learning_rate_placeholder: lr,
                                                                       phase_train_placeholder: True})
            emb_array[lab, :] = emb
        print('{:.3f}'.format(time.time() - start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class,
                                                                    image_paths, args.people_per_batch, args.alpha)
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = ({}, {}): time={:.3f}'.format(nrof_random_negs, nrof_triplets, selection_time))

        # Perform training on the selected triplets
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))

        sess.run(enqueue_op, feed_dict={image_paths_placeholder: triplet_paths_array,
                                        labels_placeholder: labels_array})

        nrof_batches = int(np.ceil(nrof_triplets * 3 / args.batch_size))
        nrof_examples = len(triplet_paths)
        # train_time = 0

        # emb_array = np.zeros((nrof_examples, embedding_size))
        # loss_array = np.zeros((nrof_triplets,))
        summary = tf.Summary()
        # step = 0

        for i in range(nrof_batches):
            start_time = time.time()
            batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size,
                         learning_rate_placeholder: lr,
                         phase_train_placeholder: True}
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)

            # emb_array[lab, :] = emb
            # loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [{}][{}/{}] Time: {:.3f} Loss: {:.5f}'.format(epoch, batch_number + 1, args.epoch.size, duration, err))
            # batch_number += 1

            # train_time += duration
            summary.value.add(tag='loss', simple_value=err)

        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
    return True


def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in range(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in range(j, nrof_images):  # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = dataset.nrof_classes
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = dataset.classes[class_index].nrof_images
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset.classes[class_index].files[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame,
             batch_size,
             nrof_folds, log_dir, step, summary_writer, embedding_size):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on LFW images: ', end='')

    nrof_images = len(actual_issame) * 2
    assert (len(image_paths) == nrof_images)
    labels_array = np.reshape(np.arange(nrof_images), (-1, 3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(np.ceil(nrof_images / batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        batch_size = min(nrof_images - i * batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                   learning_rate_placeholder: 0.0,
                                                                   phase_train_placeholder: False})
        emb_array[lab, :] = emb
        label_check_array[lab] = 1
    print('%.3f' % (time.time() - start_time))

    assert (np.all(label_check_array == 1))

    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'lfw_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))


if __name__ == '__main__':
    main()
