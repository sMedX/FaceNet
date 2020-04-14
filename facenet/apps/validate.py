"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2019 SMedX

import click
import time
import numpy as np
from pathlib import Path

from facenet import dataset, config, statistics, facenet, ioutils

DefaultConfig = config.DefaultConfig()


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options for the application.')
def main(**args_):
    start_time = time.monotonic()

    args = config.YAMLConfig(args_['config'])
    np.random.seed(args.seed)

    dbase = dataset.DBase(args.dataset)
    print(dbase)

    emb = facenet.Embeddings(dbase, args)
    print(emb)

    validate = statistics.FaceToFaceValidation(emb.embeddings, dbase.labels, args.validation)
    validate.evaluate()
    validate.write_report(args.report, info=dbase.__repr__() + emb.__repr__())
    print(validate)

    ioutils.write_elapsed_time(validate.file, start_time)


if __name__ == '__main__':
    main()
