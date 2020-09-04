"""Imports a model metagraph and checkpoint file, converts the variables to constants
and exports the model as a graphdef protobuf
"""
# MIT License
# Copyright (c) 2020 sMedX

import click
from pathlib import Path

from facenet import tfutils, config, facenet


@click.command()
@click.option('--model_dir', default=config.default_model, type=Path,
              help='Directory with the meta graph and checkpoint files containing model parameters.')
def main(**args):
    files = config.data_dir.glob('*' + config.file_extension)
    images = [facenet.load_images(f) for f in files]

    h5file = tfutils.export_h5(args['model_dir'])


if __name__ == '__main__':
    main()
