
import os
import sys
from subprocess import Popen, PIPE
import numpy as np
from scipy import spatial
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw

from facenet import ioutils


def file2text(file):
    return os.path.join(os.path.basename(os.path.dirname(file)), os.path.splitext(os.path.basename(file))[0])


def generate_filename(dirname, value, file1, file2):
    dir1 = os.path.basename(os.path.dirname(file1))
    name1 = os.path.splitext(os.path.basename(file1))[0]

    dir2 = os.path.basename(os.path.dirname(file2))
    name2 = os.path.splitext(os.path.basename(file2))[0]

    if dir1 == dir2:
        s = os.path.join(dirname, '{}|{} & {} & {:2.3f}.png'.format(dir1, name1, name2, value))
    else:
        s = os.path.join(dirname, '{}|{} & {}|{} & {:2.3f}.png'.format(dir1, name1, dir2, name2, value))
    return s


class ConcatenateImages:
    def __init__(self, file1, file2, distance, font_size=13):
        self.file1 = file1
        self.file2 = file2
        self.distance = distance

        img1 = ioutils.read_image(file1)
        img2 = ioutils.read_image(file2)
        self.img = Image.fromarray(np.concatenate([np.array(img1), np.array(img2)], axis=1))

        text = '{} & {}\n{:2.3f}'.format(file2text(file1), file2text(file2), distance)

        if sys.platform == 'win32':
            font = ImageFont.truetype("arial.ttf", font_size)
        else:
            font = ImageFont.truetype("LiberationSans-Regular.ttf", font_size)

        draw = ImageDraw.Draw(self.img)
        draw.text((0, 0), text, (0, 255, 0), font=font)

    def save(self, outdir):
        filename = generate_filename(outdir, self.distance, self.file1, self.file2)
        ioutils.write_image(self.img, filename)


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


def int64_feature(value):
    """Wrapper for insert int64 feature into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for insert float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for insert bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def write_tfrecord(tfrecord, files, labels, embeddings):

    with tf.python_io.TFRecordWriter(tfrecord) as tfwriter:
        for i, (file, label, embedding) in enumerate(zip(files, labels, embeddings)):
            add_to_tfrecord(tfwriter, file.encode(), label, embedding.tolist())

            if (i+1) % 1000 == 0:
                print('\r{}/{} samples have been added to tfrecord file.'.format(i+1, len(files)), end='')

    print('\rtfrecord file {} has been written, number of samples is {}.'.format(tfrecord, len(files)))


def add_to_tfrecord(tfwriter, file, label, embedding):
    example = tf.train.Example(
        features=tf.train.Features(feature={
            'filename': bytes_feature(file),
            'label': int64_feature(label),
            'embedding': float_feature(embedding)
            }))

    tfwriter.write(example.SerializeToString())


def read_tfrecord(tfrecord, mode='array'):
    tfrecord = os.path.expanduser(tfrecord)
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord)

    files = []
    labels = []
    embeddings = []

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        feature = example.features.feature

        file = feature['filename'].bytes_list.value[0]
        files.append(file.decode())

        label = feature['label'].int64_list.value[0]
        labels.append(label)

        embedding = feature['embedding'].float_list.value
        embeddings.append(embedding)

    embeddings = np.array(embeddings)

    return files, labels, embeddings


class TFRecord:
    def __init__(self, tffile):
        self.tffile = tffile
        self.files, self.labels, self.embeddings = read_tfrecord(self.tffile)

    def __repr__(self):
        """Representation of the database"""
        info = ('class {}\n'.format(self.__class__.__name__) +
                'TFReccord {}\n'.format(self.tffile) +
                'Embeddings [{}, {}]\n'.format(self.embeddings.shape[0], self.embeddings.shape[1]))
        return info
