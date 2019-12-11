"""Imports a model metagraph and checkpoint file, converts the variables to constants
and exports the model as a graphdef protobuf
"""
# MIT License
# 
# Copyright (c) 2019 sMedX

import click
import pathlib
from facenet import facenet


@click.command()
@click.option('--model_dir', type=pathlib.Path,
              help='Directory with the meta graph and checkpoint files containing model parameters')
def main(**args):
    model_dir = args['model_dir'].expanduser()
    facenet.save_freeze_graph(model_dir=model_dir)


if __name__ == '__main__':
    main()
