# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib

src_dir = plib.Path(__file__).parents[1]
file_extension = '.png'


class DefaultConfig:
    def __init__(self):
        self.model = src_dir.joinpath('models', '20190727-080213')
        self.pretrained_checkpoint = src_dir.joinpath('models', '20190727-080213', 'model-20190727-080213.ckpt-275')

        # type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance
        self.distance_metric = 0

        # image size (height, width) in pixels
        self.image_size = 160

        # embedding size
        self.embedding_size = 512

        # image standardisation
        # False: tf.image.per_image_standardization(image)
        # True: (tf.cast(image, tf.float32) - 127.5) / 128.0
        self.image_standardization = True
