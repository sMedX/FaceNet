"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import ReLU, Conv2D, MaxPool2D, AvgPool2D, Dense, Flatten, Dropout, BatchNormalization


default_config = {
    'reduction_a': {
        'filters': [[384], [192, 192, 256]]
    },
    'reduction_b': {
        'filters': [[256, 384], [256, 256], [256, 256, 256]]
    },
    'block35': {
        'repeat': 5,
        'scale': 0.17,
        'activation': 'relu'
    },
    'block17': {
        'repeat': 10,
        'scale': 0.10,
        'activation': 'relu'
    },
    'block8': [
        {
            'repeat': 5,
            'scale': 0.2,
            'activation': 'relu'
        },
        {
            'scale': 1.0,
            'activation': None
        }
    ],
    'features': {
        'size': 512,
        'dropout_rate': 0.5
    },
}

kernel_initializer = tf.keras.initializers.GlorotNormal()

batch_normalization = {
    # Decay for the moving averages.
    'momentum': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    'fused': False,
    'trainable': True,
    'center': True,
    'scale': False
}


# Inception-Resnet-A
class Block35(keras.layers.Layer):
    """Builds the 35x35 resnet block."""
    def __init__(self, config):
        super().__init__()
        if config.get('batch_normalization') is None:
            config['batch_normalization'] = batch_normalization
        self.config = config
        self.activation = tf.keras.activations.deserialize(config['activation'])

        self.tower_conv = tf.keras.Sequential([
            Conv2D(32, 1, strides=1, padding='same', activation='relu', name='Conv2d_1x1',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU()
        ])

        self.tower_conv1 = tf.keras.Sequential([
            Conv2D(32, 1, strides=1, padding='same', activation=None, name='Conv2d_0a_1x1',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(),
            ReLU(),
            Conv2D(32, 3, strides=1, padding='same', activation=None, name='Conv2d_0b_3x3',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU()
        ])

        self.tower_conv2 = tf.keras.Sequential([
            Conv2D(32, 1, strides=1, padding='same', activation=None, name='Conv2d_0a_1x1',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            Conv2D(32, 3, strides=1, padding='same', activation=None, name='Conv2d_0b_3x3',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(),
            ReLU(),
            Conv2D(32, 3, strides=1, padding='same', activation=None, name='Conv2d_0c_3x3',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU()
        ])

        self.up = None

    def build(self, input_shape):
        self.up = Conv2D(input_shape[-1], 1, strides=1, padding='same', activation='relu', name='Conv2d_1x1',
                         kernel_initializer=kernel_initializer
                         )

    def call(self, net, **kwargs):
        values = [self.tower_conv(net), self.tower_conv1(net), self.tower_conv2(net)]
        mixed = tf.concat(values, 3)

        net += self.config['scale'] * self.up(mixed)

        if self.activation:
            net = self.activation(net)

        return net


class Block17(keras.layers.Layer):
    """Builds the 17x17 resnet block."""
    def __init__(self, config):
        super().__init__()
        if config.get('batch_normalization') is None:
            config['batch_normalization'] = batch_normalization
        self.config = config
        self.activation = tf.keras.activations.deserialize(config['activation'])

        self.tower_conv = tf.keras.Sequential([
            Conv2D(128, 1, strides=1, padding='same', activation='relu', name='Conv2d_1x1',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU()
        ])

        self.tower_conv1 = tf.keras.Sequential([
            Conv2D(128, 1, strides=1, padding='same', activation=None, name='Conv2d_0a_1x1',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            Conv2D(128, (1, 7), strides=1, padding='same', activation=None, name='Conv2d_0b_1x7',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(),
            ReLU(),
            Conv2D(128, (7, 1), strides=1, padding='same', activation=None, name='Conv2d_0c_7x1',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU()
        ])

        self.up = None

    def build(self, input_shape):
        self.up = Conv2D(input_shape[-1], 1, strides=1, padding='same', activation='relu', name='Conv2d_1x1',
                         kernel_initializer=kernel_initializer
                         )

    def call(self, net, **kwargs):
        mixed = tf.concat([self.tower_conv(net), self.tower_conv1(net)], 3)
        net += self.config['scale'] * self.up(mixed)

        if self.activation:
            net = self.activation(net)

        return net


# Inception-Resnet-C
class Block8(keras.layers.Layer):
    """Builds the 8x8 resnet block."""
    def __init__(self, config):
        super().__init__()
        if config.get('batch_normalization') is None:
            config['batch_normalization'] = batch_normalization
        self.config = config
        self.activation = tf.keras.activations.deserialize(config['activation'])

        self.tower_conv = tf.keras.Sequential([
            Conv2D(192, 1, strides=1, padding='same', activation=None, name='Conv2d_1x1',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU()
        ])

        self.tower_conv1 = tf.keras.Sequential([
            Conv2D(192, 1, strides=1, padding='same', activation=None, name='Conv2d_0a_1x1',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(),
            ReLU(),
            Conv2D(192, (1, 3), strides=1, padding='same', activation=None, name='Conv2d_0b_1x3',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            Conv2D(192, (3, 1), strides=1, padding='same', activation=None, name='Conv2d_0c_3x1',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(),
            ReLU()
        ])

        self.up = None

    def build(self, input_shape):
        self.up = Conv2D(input_shape[-1], 1, strides=1, padding='same', activation='relu', name='Conv2d_1x1',
                         kernel_initializer=kernel_initializer
                         )

    def call(self, net, **kwargs):
        mixed = tf.concat([self.tower_conv(net), self.tower_conv1(net)], 3)
        net += self.config['scale'] * self.up(mixed)

        if self.activation:
            net = self.activation(net)

        return net


class ReductionA(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        if config.get('batch_normalization') is None:
            config['batch_normalization'] = batch_normalization
        self.config = config

        filters = config['filters'][0]
        self.tower_conv = tf.keras.Sequential([
            Conv2D(filters[0], 3, strides=2, padding='valid', activation=None, name='Conv2d_1a_3x3',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU()
        ])

        filters = config['filters'][1]
        self.tower_conv1 = tf.keras.Sequential([
            Conv2D(filters[0], 1, strides=1, padding='same', activation=None, name='Conv2d_0a_1x1',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            Conv2D(filters[1], 3, strides=1, padding='same', activation=None, name='Conv2d_0b_3x3',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            Conv2D(filters[2], 3, strides=2, padding='valid', activation=None, name='Conv2d_1a_3x3',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU()
        ])

        self.tower_pool = MaxPool2D(3, strides=2, padding='valid', name='MaxPool_1a_3x3')

    def call(self, net, **kwargs):
        net = tf.concat([self.tower_conv(net), self.tower_conv1(net), self.tower_pool(net)], 3)
        return net


class ReductionB(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        if config.get('batch_normalization') is None:
            config['batch_normalization'] = batch_normalization
        self.config = config

        filters = config['filters'][0]
        self.tower_conv = tf.keras.Sequential([
            Conv2D(filters[0], 1, strides=1, padding='same', activation=None, name='Conv2d_0a_1x1',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            Conv2D(filters[1], 3, strides=2, padding='valid', activation=None, name='Conv2d_1a_3x3',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU()
        ])

        filters = config['filters'][1]
        self.tower_conv1 = tf.keras.Sequential([
            Conv2D(filters[0], 1, strides=1, padding='same', activation=None, name='Conv2d_0a_1x1',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            Conv2D(filters[1], 3, strides=2, padding='valid', activation=None, name='Conv2d_1a_3x3',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU()
        ])

        filters = config['filters'][2]
        self.tower_conv2 = tf.keras.Sequential([
            Conv2D(filters[0], 1,  strides=1, padding='same', activation=None, name='Conv2d_0a_1x1',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            Conv2D(filters[1], 3,  strides=1, padding='same', activation=None, name='Conv2d_0b_3x3',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            Conv2D(filters[2], 3, strides=2, padding='valid', activation=None, name='Conv2d_1a_3x3',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU()
        ])

        self.tower_pool = MaxPool2D(3, strides=2, padding='valid', name='MaxPool_1a_3x3')

    def call(self, net, **kwargs):
        values = [self.tower_conv(net), self.tower_conv1(net), self.tower_conv2(net), self.tower_pool(net)]
        net = tf.concat(values, 3)

        return net


class Features(keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        if config.get('batch_normalization') is None:
            config['batch_normalization'] = batch_normalization
        self.config = config

        self.flatten = Flatten()
        self.batch_normalization = BatchNormalization(**self.config['batch_normalization'])
        self.dense = Dense(config['size'], activation=None, name='logits',
                           kernel_initializer=kernel_initializer
                           )

    def call(self, inputs, **kwargs):
        outputs = self.flatten(inputs)
        outputs = self.batch_normalization(outputs)
        outputs = self.dense(outputs)
        return outputs


class InceptionResnetV1(keras.Model):
    def __init__(self, image_processing,
                 config=None):
        super().__init__()

        if config is None:
            config = default_config
        if config.get('batch_normalization') is None:
            config['batch_normalization'] = batch_normalization
        self.config = config

        self.image_processing = image_processing

        self.conv2d = tf.keras.Sequential([
            Conv2D(32, 3, strides=2, padding='valid', use_bias=False, activation=None, name='Conv2d_1a_3x3',
                   kernel_initializer=kernel_initializer,
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            Conv2D(32, 3, strides=1, padding='valid', use_bias=False, activation=None, name='Conv2d_2a_3x3',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            Conv2D(64, 3, strides=1, padding='valid', use_bias=False, activation=None, name='Conv2d_2b_3x3',
                   kernel_initializer=kernel_initializer
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            MaxPool2D(3, strides=2, padding='valid', name='MaxPool_3a_3x3'),
            Conv2D(80, 1, strides=1, padding='valid', use_bias=False, activation=None, name='Conv2d_3b_1x1',
                   kernel_initializer=kernel_initializer,
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            Conv2D(192, 3, strides=1, padding='valid', use_bias=False, activation=None, name='Conv2d_4a_3x3',
                   kernel_initializer=kernel_initializer,
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU(),
            Conv2D(256, 3, strides=2, padding='valid', use_bias=False, activation=None, name='Conv2d_4b_3x3',
                   kernel_initializer=kernel_initializer,
                   ),
            BatchNormalization(**self.config['batch_normalization']),
            ReLU()
        ])

        # repeat block35
        layers = [Block35(config=config['block35']) for _ in range(config['block35']['repeat'])]
        self.repeat_block35 = tf.keras.Sequential(layers=layers, name='block35')

        self.reduction_a = ReductionA(config['reduction_a'])

        # repeat block17
        layers = [Block17(config=config['block17']) for _ in range(config['block17']['repeat'])]
        self.repeat_block17 = tf.keras.Sequential(layers=layers, name='block17')

        self.reduction_b = ReductionB(config['reduction_b'])

        # repeat block8
        conf = config['block8'][0]
        layers = [Block8(config=conf) for _ in range(conf['repeat'])]
        self.repeat_block8 = tf.keras.Sequential(layers=layers, name='block8')

        conf = config['block8'][1]
        self.block8 = Block8(config=conf)

        self.avg_pool2d = AvgPool2D([3, 3], padding='VALID', name='AvgPool_1a_8x8')

        self.features = Features(config['features'])

    def call(self, inputs, **kwargs):
        outputs = self.image_processing(inputs)

        outputs = self.conv2d(outputs)

        # 5 x Inception-resnet-A
        outputs = self.repeat_block35(outputs)

        outputs = self.reduction_a(outputs)

        # 10 x Inception-Resnet-B
        outputs = self.repeat_block17(outputs)

        # Reduction-B
        outputs = self.reduction_b(outputs)

        # 5 x Inception-Resnet-C
        outputs = self.repeat_block8(outputs)

        outputs = self.block8(outputs)

        outputs = self.avg_pool2d(outputs)

        outputs = self.features(outputs)

        return outputs

    def embedding(self, images):
        output = self(images, training=False)
        output = tf.nn.l2_normalize(output, 1, 1e-10, name='embedding')
        return output

