"""Faster-RCNN with Inception (v3)"""

import os
import tensorflow as tf
import numpy as np

default_weights = os.path.join(os.path.dirname(__file__), 'weights', 'frozen_inference_graph.pb')


def load_graph(filename):
    print('Load model from: {}'.format(filename))

    with tf.gfile.GFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='')
    return graph


class FaceDetector:
    def __init__(self, weights_file=default_weights, gpu_memory_fraction=1.0, threshold=0.7):
        graph = load_graph(weights_file)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)

        self.sess = tf.Session(graph=graph,
                               config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        
        self.image_tensor = graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = graph.get_tensor_by_name('detection_scores:0')

        # The following processing is only for single image
        self.detection_boxes = tf.squeeze(self.detection_boxes)
        self.detection_scores = tf.squeeze(self.detection_scores)

        self.threshold = threshold

    def get_faces(self, im):
        boxes, scores = self.sess.run([self.detection_boxes, self.detection_scores],
                                      feed_dict={self.image_tensor: np.expand_dims(im, 0)})

        indexes = scores > self.threshold

        boxes = boxes[indexes, :]
        scores = scores[indexes]

        boxes[:, [0, 2]] *= im.shape[0]
        boxes[:, [1, 3]] *= im.shape[1]

        return boxes, scores
