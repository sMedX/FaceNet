
import numpy as np
import torch

import tensorflow as tf
import tensorflow.contrib.slim as slim

inp_channels = 3
out_channels = 32
kernel_size = (3, 3)
stride = 2
slim_padding = 'valid'

height = 160
width = 160


def apply_slim_conv2d(inputs):
    """
    # https://github.com/google-research/tf-slim/blob/master/tf_slim/layers/layers.py#L1152

    :param inputs:
    :return:
    """
    inputs = tf.convert_to_tensor(inputs)
    net = slim.conv2d(inputs, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=slim_padding)

    with tf.Session() as sess:
        sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
        out = sess.run(net)

        variables = []
        for var in tf.compat.v1.trainable_variables():
            print(var)
            variables.append(sess.run(var))

        return out, variables


def apply_torch_conv2d(inputs, variables):
    if slim_padding == 'same':
        torch_padding = [s // 2 for s in kernel_size]
    elif slim_padding == 'valid':
        torch_padding = 0
    else:
        raise ValueError()

    inputs_torch = torch.from_numpy(np.transpose(inputs, axes=[0, 3, 1, 2]))
    conv2d = torch.nn.Conv2d(inp_channels, out_channels, kernel_size=kernel_size, stride=stride,
                             padding=torch_padding, padding_mode='zeros', bias=False)

    weight = np.transpose(variables[0], axes=[3, 2, 0, 1])
    conv2d.weight = torch.nn.Parameter(torch.from_numpy(weight))

    out = conv2d.forward(inputs_torch)
    out = torch.nn.ReLU()(out)

    out = out.detach().cpu().numpy()
    out = np.transpose(out, axes=[0, 2, 3, 1])

    return out


def main(**options):
    np.random.seed(0)
    tf.set_random_seed(0)

    inputs = np.float32(2 * np.random.rand(1, height, width, inp_channels) - 1)

    print('------------------------------')
    out1, variables = apply_slim_conv2d(inputs)
    print(out1.shape)

    print('------------------------------')
    out2 = apply_torch_conv2d(inputs, variables)
    print(out2.shape)

    print(out1[0, :, :, 1])
    print(out2[0, :, :, 1])

    norm = np.max(np.abs(out1 - out2))
    print('maximum error', norm)


if __name__ == '__main__':
    main()


