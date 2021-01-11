"""Exports trainable parameters to h5 file
"""
# MIT License
# Copyright (c) 2020 sMedX

import click
from pathlib import Path

from facenet import tfutils, config
import facenet.models.inception_resnet_v1 as module


@click.command()
@click.option('--model_dir', default=config.default_model_path, type=Path,
              help='Directory with the meta graph and checkpoint files containing model parameters.')
def main(**options):

    tfutils.export_h5(options['model_dir'], module)


if __name__ == '__main__':
    main()
