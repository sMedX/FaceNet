# coding:utf-8
"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
# 
# Copyright (c) 2020 sMedX
# 
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit

import click
from pathlib import Path

import tensorflow as tf

from facenet.models.inception_resnet_v1_tf2 import InceptionResnetV1 as FaceNet
from facenet import statistics, config, dataset
from facenet import config_tf2 as config
from facenet import facenet_tf2 as facenet
from facenet.logging import configure_logging


@click.command()
@click.option('--config', default=None, type=Path,
              help='Path to yaml config file with used options of the application.')
def main(**options):
    cfg = config.train_softmax(options)
    configure_logging(cfg.logs)

    # ------------------------------------------------------------------------------------------------------------------
    # define train and test datasets
    loader = facenet.ImageLoader(config=cfg.image)

    train_dbase = dataset.DBase(cfg.dataset)
    train_dataset = facenet.dataset(train_dbase.files, train_dbase.labels, loader,
                                    shuffle=True, config=cfg)

    test_dbase = dataset.DBase(cfg.validate.dataset)
    test_dataset = facenet.dataset(test_dbase.files, test_dbase.labels, loader,
                                   shuffle=False, config=cfg)

    # ------------------------------------------------------------------------------------------------------------------
    # import network
    inputs = facenet.inputs(cfg.image)

    model = FaceNet(input_shape=facenet.inputs(cfg.image),
                    image_processing=facenet.ImageProcessing(cfg.image))
    model.summary()

    # define model to train
    kernel_regularizer = tf.keras.regularizers.deserialize(model.config.regularizer.kernel.as_dict)

    network = tf.keras.Sequential([
        model,
        tf.keras.layers.Dense(train_dbase.nrof_classes,
                              activation=None,
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              kernel_regularizer=kernel_regularizer,
                              bias_initializer='zeros',
                              bias_regularizer=None,
                              name='logits')
    ])

    if cfg.model.checkpoint:
        print(f'Restore checkpoint {cfg.model.checkpoint}')
        network.load_weights(cfg.model.checkpoint)

    # ------------------------------------------------------------------------------------------------------------------
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cfg.model.path / 'model',
                                                             save_weights_only=True,
                                                             verbose=1)

    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(
        facenet.LearningRateScheduler(config=cfg.train.learning_rate),
        verbose=True
    )

    network.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer=tf.keras.optimizers.Adam())

    network.fit(
        train_dataset,
        epochs=cfg.train.max_nrof_epochs,
        steps_per_epoch=None,
        callbacks=[
            checkpoint_callback,
            learning_rate_callback
        ]
    )
    network.save(cfg.model.path / 'model')

    embeddings, labels = facenet.evaluate_embeddings(model, test_dataset)
    statistics.FaceToFaceValidation(embeddings, labels, cfg.validate.validate)

    print(f'Model and logs have been saved to the directory: {cfg.model.path}')


if __name__ == '__main__':
    main()

