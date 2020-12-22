# coding:utf-8
"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
# 
# Copyright (c) 2020 sMedX
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

# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit

import click
import time
from tqdm import tqdm
from loguru import logger
from pathlib import Path
import itertools

import tensorflow as tf

from facenet.models.inception_resnet_v1_tf2 import InceptionResnetV1 as FaceNet
from facenet import ioutils, statistics, config, dataset, facenet
from facenet import config_tf2 as config
from facenet import facenet_tf2 as facenet
from facenet.logging import configure_logging


@click.command()
@click.option('--config', default=None, type=Path,
              help='Path to yaml config file with used options of the application.')
def main(**options):
    app_file_name = 'train_softmax'

    cfg = config.train_softmax(app_file_name, options)
    configure_logging(cfg.logs)

    # ------------------------------------------------------------------------------------------------------------------
    # define train and test datasets
    loader = facenet.ImageLoader(config=cfg.image)

    train_dbase = dataset.DBase(cfg.dataset)
    train_dataset = facenet.make_train_dataset(train_dbase, loader, cfg)

    test_dbase = dataset.DBase(cfg.validate.dataset)
    test_dataset = facenet.make_test_dataset(test_dbase, loader, cfg)

    # ------------------------------------------------------------------------------------------------------------------
    # import network
    inputs = facenet.inputs(cfg.image)

    model = FaceNet(input_shape=facenet.inputs(cfg.image),
                    image_processing=facenet.ImageProcessing(cfg.image))

    # define model to train
    kernel_regularizer = tf.keras.regularizers.deserialize(model.config.regularizer.kernel.as_dict)

    network = tf.keras.Sequential([
        model,
        tf.keras.layers.Dense(train_dbase.nrof_classes,
                              activation=None,
                              kernel_initializer=tf.keras.initializers.GlorotNormal(),
                              kernel_regularizer=kernel_regularizer,
                              bias_initializer='zeros',
                              bias_regularizer=None,
                              name='logits')
    ])
    network(inputs)
    model.summary()

    # ------------------------------------------------------------------------------------------------------------------
    # learning_rate = facenet.learning_rate_schedule(cfg.train)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    network.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer=optimizer)

    network.fit(train_dataset,
                batch_size=None,
                epochs=cfg.train.epoch.nrof_epochs,
                steps_per_epoch=None)
    network.save(cfg.model.path / 'model')

    print(f'Model and logs have been saved to the directory: {cfg.model.path}')

    embeddings, labels = facenet.evaluate_embeddings(model, test_dataset)
    validation = statistics.FaceToFaceValidation(embeddings, labels, cfg.validate.validate)
    ioutils.write_text_log(cfg.logfile, validation)
    print(validation)


if __name__ == '__main__':
    main()

