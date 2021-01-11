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
@click.option('--config', default=None, type=Path,
              help='Path to yaml config file with used options for the application.')
def main(**options):
    options = config.validate(__file__, options)
    options.model.normalize = True

    dbase = dataset.DBase(options.dataset)
    ioutils.write_text_log(options.logfile, dbase)
    print(dbase)

    embeddings = facenet.EvaluationOfEmbeddings(dbase, options)
    ioutils.write_text_log(options.logfile, embeddings)
    print(embeddings)

    validate = statistics.FaceToFaceValidation(embeddings.embeddings, dbase.labels, options.validate)
    ioutils.write_text_log(options.logfile, validate)
    print(validate)

    ioutils.write_elapsed_time(options.logfile, start_time)
    print('Report has been written to the file', options.logfile)


if __name__ == '__main__':
    main()
