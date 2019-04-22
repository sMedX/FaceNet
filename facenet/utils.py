
import os
from subprocess import Popen, PIPE
import numpy as np
from scipy import spatial


def label_array(labels):

    if isinstance(labels, (np.ndarray, list)) is False:
        raise ValueError('label_array: input labels must be list or numpy.ndarray')

    if isinstance(labels, list):
        labels = np.array([labels]).transpose()

    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=0).transpose()

    labels = spatial.distance.pdist(labels, metric='sqeuclidean')
    labels = np.array(labels < 0.5, np.uint8)

    return labels


def end(start, stop):
    return '\n' if (start+1) == stop else ''


def git_hash():
    src_path, _ = os.path.split(os.path.realpath(__file__))

    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' + e.strerror

    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' + e.strerror

    return git_hash, git_diff

    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('tensorflow version: %s\n--------------------\n' % tf.__version__)  # @UndefinedVariable
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)

