"""Imports a model metagraph and checkpoint file, converts the variables to constants
and exports the model as a graphdef protobuf
"""
# MIT License
# 
# Copyright (c) 2020 sMedX

import click
import pathlib
from facenet import tfutils, config


@click.command()
@click.option('--model_dir', default=config.default_model, type=pathlib.Path,
              help='Directory with the meta graph and checkpoint files containing model parameters')
def main(**args):
    model_dir = args['model_dir'].expanduser()
    tfutils.save_freeze_graph(model_dir=model_dir)


if __name__ == '__main__':
    main()
