"""Performs face alignment and calculates L2 distance between the embeddings of images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import random
from skimage import io, transform
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
from facenet.align import detect_face
from facenet import utils, facenet

minsize = 20  # minimum size of face
threshold = (0.6, 0.7, 0.7)  # three steps's threshold
factor = 0.709  # scale factor


def main(args):

    list_of_files = args.image_files

    image_dir = os.path.expanduser(args.image_dir)
    if os.path.isdir(image_dir):
        print('directory with images: {}'.format(image_dir))
        list_of_files = utils.get_files(image_dir)

        if args.nrof_images > 0:
            random.seed(0)
            list_of_files = random.sample(list_of_files, args.nrof_images)

    images1, images2 = load_and_align_data_with_processing(list_of_files,
                                                           args.image_size, args.margin,
                                                           args.gpu_memory_fraction, args)
    distancies = []

    with tf.Graph().as_default():
        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images1, phase_train_placeholder: False}
            emb1 = sess.run(embeddings, feed_dict=feed_dict)

            feed_dict = {images_placeholder: images2, phase_train_placeholder: False}
            emb2 = sess.run(embeddings, feed_dict=feed_dict)

            print('Distancies')
            print('')
            for i, file in enumerate(list_of_files):
                dist = np.sqrt(np.sum(np.square(np.subtract(emb1[i, :], emb2[i, :]))))
                distancies.append(dist)
                print('{})  {:1.4f} {} '.format(i, dist, file))

    print('\n')
    print('number of images', len(list_of_files))
    print('statistical metrics for distances (rotation {} degrees)'.format(args.rotation))
    print('')
    print('minimal value', min(distancies))
    print(' median value', np.median(distancies))
    print('   mean value', np.mean(distancies))
    print('maximal value', max(distancies))

            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths = copy.copy(image_paths)
    img_list = []

    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove", image)
            continue

        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)

    images = np.stack(img_list)

    return images


def load_and_align_data_with_processing(image_paths, image_size, margin, gpu_memory_fraction, args):
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    print('Parameters for image processing')
    print('rotation angle in degrees', args.rotation)

    tmp_image_paths = copy.copy(image_paths)

    img_list1 = []
    img_list2 = []

    for image in tmp_image_paths:
        img = io.imread(os.path.expanduser(image))

        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove", image)
            continue

        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')

        prewhitened = facenet.prewhiten(aligned)
        img_list1.append(prewhitened)

        # rotate image
        rotated = transform.rotate(aligned, angle=args.rotation, order=1, resize=False, mode='edge', preserve_range=True)
        prewhitened = facenet.prewhiten(rotated)
        img_list2.append(prewhitened)

    images1 = np.stack(img_list1)
    images2 = np.stack(img_list2)

    return images1, images2


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_dir', type=str,
        help='Path to the data directory containing images.', default=None)
    parser.add_argument('--image_files', type=str, nargs='+',
        help='Images to compare', default=None)
    parser.add_argument('--nrof_images', type=int,
        help='Number of images to evaluete statistics.', default=0)
    parser.add_argument('--rotation', type=float,
        help='Rotation angle in degrees.', default=5)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    main(parse_arguments(sys.argv))
