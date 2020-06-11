"""Validate a face recognizer.
"""
# MIT License
# 
# Copyright (c) 2020 SMedX

import click
from pathlib import Path

from facenet import dataset, config, statistics, facenet, ioutils

start_time = ioutils.get_time()


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options for the application.')
def main(**args_):
    args = config.Validate(args_)

    dbase = dataset.DBase(args.dataset)
    dbase.write_report(args.file)
    print(dbase)

    embeddings = facenet.EvaluationOfEmbeddings(dbase, args)
    embeddings.write_report(args.file)
    print(embeddings)

    validate = statistics.FaceToFaceValidation(embeddings.embeddings, dbase.labels, args.validate)
    validate.write_report(args.file)
    print(validate)

    ioutils.write_elapsed_time(args.file, start_time)
    print('Report has been written to the file', args.file)


if __name__ == '__main__':
    main()
