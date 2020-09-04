"""Imports a model metagraph and checkpoint file, converts the variables to constants
and exports the model as a graphdef protobuf
"""
# MIT License
# Copyright (c) 2020 sMedX

import click
from pathlib import Path
from facenet import tfutils, config


@click.command()
@click.option('--model_dir', default=config.default_model, type=Path,
              help='Directory with the meta graph and checkpoint files containing model parameters.')
def main(**args):
    h5_file = tfutils.export_h5(args['model_dir'])


if __name__ == '__main__':
    main()
