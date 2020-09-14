# coding:utf-8
"""Application to apply pytorch inference
"""
# MIT License
#
# Copyright (c) 2020 sMedX
#
import click
from pathlib import Path

from facenet.models import inception_resnet_v1_torch as module
from facenet import ioutils, config, h5utils


@click.command()
@click.option('--path', default=config.data_dir, type=Path,
              help='Path to directory with images.')
@click.option('--model_dir', default=config.default_model, type=Path,
              help='Path to directory with h5 file for model parameters.')
def main(**options):
    h5file = ioutils.glob_single_file(options['model_dir'], '*.h5')

    model = module.FaceNet(h5file)
    model.test()

    image = h5utils.read(h5file, 'checkpoint/image:0')[0]
    embedding = model.forward_image(image)
    print(embedding.shape)


if __name__ == '__main__':
    main()


