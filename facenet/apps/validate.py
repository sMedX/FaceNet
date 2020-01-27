"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2019 SMedX

import click
import pathlib
from facenet import dataset, config, validate, facenet

DefaultConfig = config.DefaultConfig()


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=pathlib.Path,
              help='Path to yaml config file with used options for the application.')
@click.option('--model', default=None, type=pathlib.Path,
              help='Could be either a directory containing the meta and ckpt files or a model protobuf (.pb) file')
def main(**args_):
    args = config.YAMLConfig(args_['config'])
    if args_['model'] is not None:
        args.model = args_['model']

    if args.model is None:
        args.model = DefaultConfig.model

    if args.image.size is None:
        args.image.size = DefaultConfig.image_size

    validate.validate(args)


if __name__ == '__main__':
    main()
