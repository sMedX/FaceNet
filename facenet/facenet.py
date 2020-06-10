"""Functions for building the face recognition network.
"""
# MIT License
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

import os
import sys
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from skimage import io, transform
from sklearn.model_selection import KFold
from scipy import spatial, interpolate
from tensorflow.python.training import training
from tensorflow.compat.v1 import graph_util
import random
import re
from tensorflow.python.platform import gfile
import math
from pathlib import Path

from facenet import utils, ioutils, h5utils, FaceNet, image_processing


class Placeholders:
    batch_size = None
    phase_train = None
    files = None
    labels = None
    control = None
    learning_rate = None

    def train_feed_dict(self, learning_rate, phase_train, batch_size):
        dct = {self.learning_rate: learning_rate,
               self.phase_train: phase_train,
               self.batch_size: batch_size}
        return dct

    def run_feed_dict(self, batch_size):
        dct = {self.phase_train: False,
               self.batch_size: batch_size}
        return dct

    def enqueue_feed_dict(self, files, labels, control):
        feed_dict = {self.files: files, self.labels: labels, self.control: control}
        return feed_dict


def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss

def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers

# def get_image_paths_and_labels(dataset):
#     image_paths_flat = []
#     labels_flat = []
#     for i in range(len(dataset)):
#         image_paths_flat += dataset[i].image_paths
#         labels_flat += [i] * len(dataset[i].image_paths)
#     return image_paths_flat, labels_flat

def shuffle_examples(image_paths, labels):
    shuffle_list = list(zip(image_paths, labels))
    random.shuffle(shuffle_list)
    image_paths_shuff, labels_shuff = zip(*shuffle_list)
    return image_paths_shuff, labels_shuff

def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    output = transform.rotate(image, angle=angle, order=1, resize=False, mode='edge', preserve_range=True)
    return np.array(output, dtype=image.dtype)
    # return misc.imrotate(image, angle, 'bicubic') imrotate is deprecated in SciPy 1.0.0
  
# 1: Random rotate 2: Random crop  4: Random flip  8:  Fixed image standardization  16: Flip
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16


def create_input_pipeline(input_queue, image_size, batch_size_placeholder, nrof_preprocess_threads=4):
    images_and_labels_list = []
    for _ in range(nrof_preprocess_threads):
        filenames, label, control = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, 3)
            image = tf.cond(get_control_flag(control[0], RANDOM_ROTATE),
                            lambda: tf.py_func(random_rotate_image, [image], tf.uint8),
                            lambda: tf.identity(image))
            image = tf.cond(get_control_flag(control[0], RANDOM_CROP), 
                            lambda: tf.random_crop(image, image_size + (3,)),
                            lambda: tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))
            image = tf.cond(get_control_flag(control[0], RANDOM_FLIP),
                            lambda: tf.image.random_flip_left_right(image),
                            lambda: tf.identity(image))
            image = tf.cond(get_control_flag(control[0], FIXED_STANDARDIZATION),
                            lambda: (tf.cast(image, tf.float32) - 127.5)/128.0,
                            lambda: tf.image.per_image_standardization(image))
            image = tf.cond(get_control_flag(control[0], FLIP),
                            lambda: tf.image.flip_left_right(image),
                            lambda: tf.identity(image))
            #pylint: disable=no-member
            image.set_shape(image_size + (3,))
            images.append(image)
        images_and_labels_list.append([images, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels_list, batch_size=batch_size_placeholder, 
        shapes=[image_size + (3,), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * 100,
        allow_smaller_final_batch=True)
    
    return image_batch, label_batch

def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)
  
def _add_loss_summaries(total_loss):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op


def train_op(args, total_loss, global_step, learning_rate, update_gradient_vars):

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    optimizer = args.optimizer.lower()

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'mom':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
    
        grads = opt.compute_gradients(total_loss, update_gradient_vars)
        
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    # Add histograms for trainable variables and for gradients.
    if args.log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  


def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  

def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
  

def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = io.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images


def get_label_batch(label_data, batch_size, batch_index):
    nrof_examples = np.size(label_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = label_data[j:j+batch_size]
    else:
        x1 = label_data[j:nrof_examples]
        x2 = label_data[0:nrof_examples-j]
        batch = np.vstack([x1,x2])
    batch_int = batch.astype(np.int64)
    return batch_int


def get_batch(image_data, batch_size, batch_index):
    nrof_examples = np.size(image_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = image_data[j:j+batch_size,:,:,:]
    else:
        x1 = image_data[j:nrof_examples,:,:,:]
        x2 = image_data[0:nrof_examples-j,:,:,:]
        batch = np.vstack([x1,x2])
    batch_float = batch.astype(np.float32)
    return batch_float


def get_triplet_batch(triplets, batch_index, batch_size):
    ax, px, nx = triplets
    a = get_batch(ax, int(batch_size/3), batch_index)
    p = get_batch(px, int(batch_size/3), batch_index)
    n = get_batch(nx, int(batch_size/3), batch_index)
    batch = np.vstack([a, p, n])
    return batch


def restore_checkpoint(saver, session, path):
    if path is not None:
        path = Path(path)
        print('Restoring pre-trained model: {}'.format(path))
        saver.restore(session, str(path))


def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file) or
    # if it is a protobuf file with a frozen graph

    model_exp = Path(model).expanduser()
    print('load model: {}'.format(model))

    if model_exp.is_file():
        print('Model filename: {}'.format(model_exp))
        with gfile.FastGFile(str(model_exp), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        pb_file = model_exp.joinpath(model_exp.name + '.pb')

        if pb_file.exists():
            load_model(pb_file, input_map=input_map)
        else:
            print('Model directory: {}'.format(model_exp))
            meta_file, ckpt_file = get_model_filenames(str(model_exp))
        
            print('Metagraph file: {}'.format(meta_file))
            print('Checkpoint file: {}'.format(ckpt_file))
      
            saver = tf.train.import_meta_graph(str(model_exp.joinpath(meta_file)), input_map=input_map)
            saver.restore(tf.get_default_session(), str(model_exp.joinpath(ckpt_file)))


def get_model_filenames(model_dir):
    model_dir = str(model_dir)
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist


def distance_matrix(embeddings, distance_metric=0):
    if distance_metric == 0:
        # squared Euclidian distance
        dist = spatial.distance.pdist(embeddings, metric='sqeuclidean')
        # diff = np.subtract(embeddings1, embeddings2)
        # dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dist = 1 - spatial.distance.pdist(embeddings, metric='cosine')
        dist = np.arccos(dist) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist


def roc(thresholds, embeddings, labels, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert (embeddings.shape[0] == len(labels))

    nrof_thresholds = len(thresholds)

    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros(nrof_folds)

    indices = np.arange(embeddings.shape[0])

    # compute label matrix
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        print('\rROC {}/{}'.format(fold_idx,nrof_folds), end='')
        sys.stdout.flush()

        if subtract_mean:
            mean = np.mean(embeddings[train_set], axis=0)
        else:
            mean = 0.0

        dist_train = distance_matrix(embeddings[train_set] - mean, distance_metric)
        actual_issame_train = utils.label_array(labels[train_set])
        # actual_issame_train = spatial.distance.squareform(actual_issame[np.ix_(train_set, train_set)])

        dist_test = distance_matrix(embeddings[test_set] - mean, distance_metric)
        actual_issame_test = utils.label_array(labels[test_set])
        # actual_issame_test = spatial.distance.squareform(actual_issame[np.ix_(test_set, test_set)])

        # Find the best threshold for the fold
        acc_train = np.zeros(nrof_thresholds)
        for idx, threshold in enumerate(thresholds):
            _, _, acc_train[idx] = calculate_accuracy(threshold, dist_train, actual_issame_train)

        best_threshold_index = np.argmax(acc_train)

        for idx, threshold in enumerate(thresholds):
            tprs[fold_idx, idx], fprs[fold_idx, idx], _ = calculate_accuracy(threshold, dist_test, actual_issame_test)

        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist_test, actual_issame_test)

    print()

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)

    return tpr, fpr, accuracy


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


def val(thresholds, embeddings, labels, far_target=1e-3, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert (embeddings.shape[0] == len(labels))

    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(embeddings.shape[0])

    # compute label matrix
    # actual_issame = utils.label_matrix(image_paths, diagonal=False)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        print('\rVAL {}/{}'.format(fold_idx, nrof_folds), end='')
        sys.stdout.flush()

        if subtract_mean:
            mean = np.mean(embeddings[train_set], axis=0)
        else:
            mean = 0.0

        dist_train = distance_matrix(embeddings[train_set] - mean, distance_metric)
        actual_issame_train = utils.label_array(labels[train_set])

        dist_test = distance_matrix(embeddings[test_set] - mean, distance_metric)
        actual_issame_test = utils.label_array(labels[test_set])

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for idx, threshold in enumerate(thresholds):
            _, far_train[idx] = calculate_val_far(threshold, dist_train, actual_issame_train)

        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist_test, actual_issame_test)

    print()

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)

    return val_mean, val_std, far_mean


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
    
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = 1 if n_same == 0 else float(true_accept) / float(n_same)
    far = 1 if n_diff == 0 else float(false_accept) / float(n_diff)
    return val, far


def list_variables(filename):
    reader = training.NewCheckpointReader(filename)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    return names


def put_images_on_grid(images, shape=(16,8)):
    nrof_images = images.shape[0]
    img_size = images.shape[1]
    bw = 3
    img = np.zeros((shape[1]*(img_size+bw)+bw, shape[0]*(img_size+bw)+bw, 3), np.float32)
    for i in range(shape[1]):
        x_start = i*(img_size+bw)+bw
        for j in range(shape[0]):
            img_index = i*shape[0]+j
            if img_index>=nrof_images:
                break
            y_start = j*(img_size+bw)+bw
            img[x_start:x_start+img_size, y_start:y_start+img_size, :] = images[img_index, :, :, :]
        if img_index >= nrof_images:
            break
    return img


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
    output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(","), variable_names_whitelist=whitelist_names)
    return output_graph_def


def save_variables_and_metagraph(sess, saver, model_dir, step, model_name=None):

    if model_name is None:
        model_name = model_dir.stem

    # save the model checkpoint
    # start_time = time.time()
    checkpoint_path = model_dir.joinpath('model-{}.ckpt'.format(model_name))
    saver.save(sess, str(checkpoint_path), global_step=step, write_meta_graph=False)
    # save_time_variables = time.time() - start_time
    print('saving checkpoint: {}-{}'.format(checkpoint_path, step))

    metagraph_filename = model_dir.joinpath('model-{}.meta'.format(model_name))

    if not metagraph_filename.exists():
        saver.export_meta_graph(str(metagraph_filename))
        print('saving meta graph:', metagraph_filename)


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
            tf.get_default_session().run(tf.global_variables_initializer())
            tf.get_default_session().run(tf.local_variables_initializer())
            saver.restore(tf.get_default_session(), str(model_dir.joinpath(ckpt_file)))

            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()

            # Freeze the graph def
            output_graph_def = freeze_graph_def(sess, input_graph_def, 'embeddings,label_batch')

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(str(output_file), 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('{} ops in the final graph: {}'.format(len(output_graph_def.node), str(output_file)))

    return output_file


def learning_rate_value(epoch, config):
    if config.value is not None:
        return config.value

    if epoch >= config.schedule[-1][0]:
        return None

    for (epoch_, lr_) in config.schedule:
        if epoch < epoch_:
            return lr_


class EvaluationOfEmbeddings:
    def __init__(self, dbase, config):
        self.config = config
        self.dbase = dbase

        facenet = FaceNet(self.config.model)

        self.embeddings = np.zeros([dbase.nrof_images, facenet.embedding_size])

        print('Running forward pass on images')

        for i in tqdm(range(0, self.dbase.nrof_images, self.config.batch_size)):
            files = self.dbase.files[i:i + self.config.batch_size]
            image_batch = []

            for file in files:
                img = ioutils.read_image(file)
                img = image_processing(img, config.image)
                image_batch.append(img)

            self.embeddings[i:i + config.batch_size, :] = facenet.image_to_embedding(image_batch)

    def __repr__(self):
        info = ('{}\n'.format(self.__class__.__name__) +
                'model: {}\n'.format(self.config.model) +
                'embedding size: {}\n'.format(self.embeddings.shape))
        return info

    def write_report(self, file):
        info = 64 * '-' + '\n' + str(self)
        ioutils.write_to_file(file, info, mode='a')


class Summary:
    def __init__(self, summary_writer, h5file, tag=''):
        self._summary_writer = summary_writer
        self._h5file = h5file
        self._counts = {}
        self._tag = tag

    @staticmethod
    def get_info_str(output):
        if output.get('tensor_op'):
            output = output['tensor_op']

        info = ''
        for key, item in output.items():
            info += ' {} {:.5f}'.format(key, item)

        return info[1:]

    @property
    def tag(self):
        return self._tag

    def _count(self, name):
        if name not in self._counts.keys():
            self._counts[name] = -1
        self._counts[name] += 1
        return self._counts[name]

    def write_tf_summary(self, output, tag=None):
        if output.get('summary_op'):
            self._summary_writer.add_summary(output['summary_op'], global_step=self._count('summary_op'))

        if output.get('tensor_op'):
            output = output['tensor_op']

        if tag is None:
            tag = self.tag

        summary = tf.Summary()

        def add_summary(dct, tag):
            tag = tag + '/' if tag else ''
            for key, value in dct.items():
                if isinstance(value, dict):
                    add_summary(value, tag + key)
                else:
                    summary.value.add(tag=tag + key, simple_value=value)

        add_summary(output, tag)
        self._summary_writer.add_summary(summary, global_step=self._count(tag))

    def write_h5_summary(self, output):
        h5utils.write_dict(self._h5file, output, group=self._tag)

    def write_elapsed_time(self, value):
        tag = self._tag + '/time' if self.tag else 'time'

        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self._summary_writer.add_summary(summary, global_step=self._count('elapsed_time'))

        h5utils.write(self._h5file, tag, value)

