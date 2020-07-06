# flake8: noqa
# coding:utf-8

# MIT License
# Copyright (c) 2020 sMedX

import tensorflow.compat.v1 as tf
import numpy as np
from typing import Iterable

from facenet import tfutils, ioutils


class FaceNet:
    def __init__(self, config):
        """
        import numpy as np
        from facenet import FaceNet

        facenet = FaceNet(pb_file)
        emb = facenet.image_to_embedding(np.zeros([160, 160, 3]))
        print(emb)
        """
        self._session = tf.Session()
        tfutils.load_model(config.path)

        # Get input and output tensors
        graph = tf.get_default_graph()
        self._embeddings = graph.get_tensor_by_name("embeddings:0")

        self._image_placeholder = graph.get_tensor_by_name("input:0")
        self._phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

        self._feed_dict = {
            self._image_placeholder: None,
            self._phase_train_placeholder: False
        }

    @property
    def embedding_size(self):
        return self._embeddings.shape[1]

    def evaluate(self, images):
        # Run forward pass to calculate embeddings
        self._feed_dict[self._image_placeholder] = images
        return self._session.run(self._embeddings, feed_dict=self._feed_dict)

    def image_to_embedding(self, image_arrays: Iterable[np.ndarray]) -> np.ndarray:
        image_arrays = np.asarray(image_arrays)
        if image_arrays.ndim == 3:
            image_arrays = np.expand_dims(image_arrays, 0)

        return self.evaluate(image_arrays)


def image_processing(img, config):
    img = ioutils.pil2array(img)

    if config.standardization:
        img = (img - 127.5)/128.0

    return img
