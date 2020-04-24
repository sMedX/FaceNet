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
from pathlib import Path

from facenet import dataset, config, statistics, facenet, ioutils

DefaultConfig = config.DefaultConfig()


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options for the application.')
def main(**args_):
    start_time = time.monotonic()

    args = config.Validate(args_)

    dbase = dataset.DBase(args.dataset)
    print(dbase)

    embeddings = facenet.Embeddings(dbase, args)
    print(embeddings)

    validate = statistics.FaceToFaceValidation(embeddings.data, dbase.labels, args.validate)
    validate.write_report(info=(str(dbase), str(embeddings)))
    print(validate)

    ioutils.write_elapsed_time(args.validate.file, start_time)


if __name__ == '__main__':
    main()
