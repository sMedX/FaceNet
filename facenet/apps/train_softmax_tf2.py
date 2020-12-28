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
from loguru import logger

import tensorflow as tf

from facenet.models.inception_resnet_v1_tf2 import InceptionResnetV1 as FaceNet
from facenet import statistics, config, dataset, logging
from facenet import config_tf2 as config
from facenet import facenet_tf2 as facenet


@click.command()
@click.option('--config', default=None, type=Path,
              help='Path to yaml config file with used options of the application.')
def main(**options):
    cfg = config.train_softmax(options)
    logging.configure_logging(cfg.logs)

    # ------------------------------------------------------------------------------------------------------------------
    # define train and test datasets
    loader = facenet.ImageLoader(config=cfg.image)

    train_dbase = dataset.DBase(cfg.dataset)
    train_dataset = facenet.dataset(train_dbase.files, train_dbase.labels, loader,
                                    batch_size=cfg.batch_size,
                                    buffer_size=10,
                                    drop_remainder=True)

    test_dbase = dataset.DBase(cfg.validate.dataset)
    test_dataset = facenet.dataset(test_dbase.files, test_dbase.labels, loader,
                                   batch_size=cfg.batch_size,
                                   buffer_size=None,
                                   drop_remainder=False)

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

    class ValidateCallback(tf.keras.callbacks.Callback):
        def __init__(self, model, dataset, every_n_epochs, max_nrof_epochs, config):
            super().__init__()
            self.model = model
            self.dataset = dataset
            self.config = config
            self.every_n_epochs = every_n_epochs
            self.max_nrof_epochs = max_nrof_epochs

        def on_epoch_end(self, epoch, logs=None):
            epoch1 = epoch + 1

            if epoch1 % self.every_n_epochs == 0 or epoch1 == self.max_nrof_epochs:
                logger.info(f'perform validation for epoch {epoch1}')

                embeddings, labels = facenet.evaluate_embeddings(self.model, self.dataset)
                statistics.FaceToFaceValidation(embeddings, labels, self.config.validate)

    validate = ValidateCallback(model, test_dataset,
                                every_n_epochs=cfg.validate.every_n_epochs,
                                max_nrof_epochs=cfg.train.max_nrof_epochs,
                                config=cfg.validate)

    network.fit(
        train_dataset,
        epochs=cfg.train.max_nrof_epochs,
        steps_per_epoch=None,
        callbacks=[
            checkpoint_callback,
            learning_rate_callback,
            validate,
        ]
    )
    network.save(cfg.model.path / 'model')

    print(f'Model and logs have been saved to the directory: {cfg.model.path}')


if __name__ == '__main__':
    main()

