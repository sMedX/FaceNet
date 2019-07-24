# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib

basedir = plib.Path(__file__).parents[1]


class DefaultConfig:
    def __init__(self):
        self.model = basedir.joinpath('models', '20190410-013706', '20190410-013706.pb')

        # type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance
        self.distance_metric = 1

        # image size (height, width) in pixels
        self.image_size = 160