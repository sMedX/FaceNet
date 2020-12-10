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
def main(**options):
    options = config.Validate(options)

    dbase = dataset.DBase(options.dataset)
    ioutils.write_text_log(options.txtfile, dbase)
    print(dbase)

    embeddings = facenet.EvaluationOfEmbeddings(dbase, options)
    ioutils.write_text_log(options.txtfile, embeddings)
    print(embeddings)

    validate = statistics.FaceToFaceValidation(embeddings.embeddings, dbase.labels, options.validate)
    ioutils.write_text_log(options.txtfile, validate)
    print(validate)

    ioutils.write_elapsed_time(options.file, start_time)
    print('Report has been written to the file', options.file)


if __name__ == '__main__':
    main()
