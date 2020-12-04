# flake8: noqa
# coding:utf-8

# MIT License
# Copyright (c) 2020 sMedX

from pathlib import Path

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import dtypes

import numpy as np
from typing import Iterable
from facenet import tfutils, ioutils
from facenet.config import YAMLConfig

nodes = {
    'input': {
        'name': 'input',
        'type': dtypes.uint8.as_datatype_enum
        },

    'output': {
        'name': 'embeddings',
        'type': dtypes.float32.as_datatype_enum
    },

}

config_nodes = {
    'image_size': {
        'name': 'image_size:0',
        'type': dtypes.uint8.as_datatype_enum
    }
}


class FaceNet:
    def __init__(self, config):
        """
        import numpy as np
        from facenet import FaceNet

        facenet = FaceNet(pb_file)
        emb = facenet.image_to_embedding(np.zeros([160, 160, 3]))
        print(emb)
        """
        if not config.input:
            config.input = nodes['input']['name'] + ':0'

        if not config.output:
            if config.normalize:
                config.output = nodes['output']['name'] + ':0'
            else:
                config.output = 'InceptionResnetV1/Bottleneck/BatchNorm/Reshape_1:0'

        self._session = tf.Session()
        tfutils.load_frozen_graph(config.path)

        # input and output tensors
        graph = tf.get_default_graph()
        self._phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')
        self._image_placeholder = graph.get_tensor_by_name(config.input)
        self._embeddings = graph.get_tensor_by_name(config.output)

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
