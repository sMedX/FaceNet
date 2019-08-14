# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib

basedir = plib.Path(__file__).parents[1]
file_extension = '.png'


class DefaultConfig:
    def __init__(self):
        self.model = str(basedir.joinpath('models', '20190727-080213'))

        # type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance
        self.distance_metric = 0

        # image size (height, width) in pixels
        self.image_size = 160

        # image standardisation
        # False: tf.image.per_image_standardization(image)
        # True: (tf.cast(image, tf.float32) - 127.5) / 128.0
        self.image_standardization = True
