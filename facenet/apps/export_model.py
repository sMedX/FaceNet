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
@click.option('--as_text', default=0, type=int,
              help='Writes the graph as an ASCII proto.')
@click.option('--strip', default=1, type=int,
              help='Removes unused nodes from a graph file.')
@click.option('--optimize', default=1, type=int,
              help='Applies optimize_for_inference for exported graph.')
def main(**args):
    pb_file = tfutils.save_freeze_graph(args['model_dir'],
                                        strip=args['strip'],
                                        optimize=args['optimize'],
                                        as_text=args['as_text'])


if __name__ == '__main__':
    main()
