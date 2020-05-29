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

import click
import time
import math
import numpy as np
import importlib
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

from facenet import ioutils, dataset, statistics, config, h5utils, facenet
from facenet.models import inception_resnet_v1 as network


def loss(y_true, y_pred):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                   labels=y_true,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
    return total_loss


class FaceNet(tf.keras.Model):
    def __init__(self, config):
        super(FaceNet, self).__init__()
        self.config = config
        self.model, _ = network.inference

        print('Building training graph')
        prelogits, _ = network.inference(image_batch,
                                         config=args.model.config,
                                         phase_train=placeholders.phase_train)
        embedding = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=args.loss.prelogits_norm_p, axis=1))

    def call(self, x):
        return x


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options of the application.')
@click.option('--learning_rate', default=None, type=float,
              help='Learning rate value')
def main(**args_):
    args = config.TrainOptions(args_, subdir=config.subdir())

    # import network
    print('import model {}'.format(args.model.module))
    network = importlib.import_module(args.model.module)

    dbase = dataset.DBase(args.dataset)
    ioutils.write_text_log(args.txtfile, str(dbase))
    print(dbase)

    x_train = dbase.files
    y_train = dbase.labels

    model = FaceNet(args)
    model.compile(loss=loss, optimizer='adam')
    model.fit(x_train, y_train, epochs=1000)


if __name__ == '__main__':
    main()

