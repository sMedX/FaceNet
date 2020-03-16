"""Performs processing images."""
# MIT License
#
# Copyright (c) 2019 Ruslan N. Kosarev

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys
import tensorflow as tf

import time
import datetime
import numpy as np
import sklearn
from sklearn.model_selection import KFold
from scipy import spatial, interpolate
from scipy.optimize import brentq
from pathlib import Path

from facenet import utils, ioutils


def pairwise_similarities(xa, xb=None, metric=0):

    if xb is None:
        sims = xa @ xa.transpose()
        sims = sims[np.triu_indices(sims.shape[0], k=1)]
    else:
        sims = xa @ xb.transpose()

    if metric == 0:
        # squared Euclidean distance
        sims = 2 * (1 - sims)
    elif metric == 1:
        # cosine
        sims = np.arccos(sims)
    else:
        raise ValueError('Undefined similarity metric {}'.format(metric))

    return sims


# def pairwise_distances(xa, xb=None, metric=0):
#     if metric == 0:
#         # squared Euclidean distance
#         if xb is None:
#             dist = spatial.distance.pdist(xa, metric='sqeuclidean')
#         else:
#             dist = spatial.distance.cdist(xa, xb, metric='sqeuclidean')
#     elif metric == 1:
#         # distance based on cosine similarity
#         if xb is None:
#             dist = spatial.distance.pdist(xa, metric='cosine')
#         else:
#             dist = spatial.distance.cdist(xa, xb, metric='cosine')
#         dist = np.arccos(1 - dist) / math.pi
#     else:
#         raise 'Undefined distance metric %d' % metric
#
#     return dist


def mean(x):
    return np.mean(np.array(x))


def std(x):
    return np.std(np.array(x))


def split_embeddings(embeddings, labels):
    emb_list = []
    for label in np.unique(labels):
        emb_array = embeddings[labels == label]
        emb_list.append(emb_array)
    return emb_list


class SimilarityCalculator:
    def __init__(self, embeddings, labels, metric=0):
        self.metric = metric
        self.embeddings = embeddings
        self.labels = labels
        self._embeddings = None

    def set_indices(self, indices):
        self._embeddings = split_embeddings(self.embeddings[indices], self.labels[indices])

    def evaluate(self, i, k):
        nrof_positive_class_pairs = self.nrof_classes
        nrof_negative_class_pairs = self.nrof_classes * (self.nrof_classes - 1) / 2

        if i == k:
            sims = pairwise_similarities(self._embeddings[i], metric=self.metric)
            weight = sims.size * nrof_positive_class_pairs
        else:
            sims = pairwise_similarities(self._embeddings[i], self._embeddings[k], metric=self.metric)
            weight = sims.size * nrof_negative_class_pairs

        return sims, weight

    @property
    def nrof_classes(self):
        return len(self._embeddings)

    def nrof_images(self, i):
        return self._embeddings[i].shape[0]


class ConfidenceMatrix:
    def __init__(self, calculator, threshold):

        self.threshold = np.array(threshold, ndmin=1)

        self.tp = np.zeros(self.threshold.size)
        self.tn = np.zeros(self.threshold.size)
        self.fp = np.zeros(self.threshold.size)
        self.fn = np.zeros(self.threshold.size)

        for i in range(calculator.nrof_classes):
            for k in range(i+1):
                sims, weight = calculator.evaluate(i, k)
                if sims.size < 1:
                    continue

                for n, threshold in enumerate(self.threshold):
                    count = np.count_nonzero(sims < threshold)

                    if i == k:
                        self.tp[n] += count/weight
                        self.fn[n] += (sims.size - count)/weight
                    else:
                        self.fp[n] += count/weight
                        self.tn[n] += (sims.size - count)/weight

    @property
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    @property
    def precision(self):
        i = (self.tp + self.fp) > 0
        precision = np.ones(self.threshold.size)
        precision[i] = self.tp[i] / (self.tp[i] + self.fp[i])
        return precision

    @property
    def tp_rates(self):
        # true positive rate, validation rate, sensitivity or recall
        i = (self.tp + self.fn) > 0
        tp_rates = np.ones(self.threshold.size)
        tp_rates[i] = self.tp[i] / (self.tp[i] + self.fn[i])
        return tp_rates

    @property
    def tn_rates(self):
        # true negative rate, 1 - false alarm rate, specificity
        i = (self.tn + self.fp) > 0
        tn_rates = np.ones(self.threshold.size)
        tn_rates[i] = self.tn[i] / (self.tn[i] + self.fp[i])
        return tn_rates

    @property
    def fp_rates(self):
        # false positive rate, false alarm rate
        return 1 - self.tn_rates

    @property
    def fn_rates(self):
        # false negative rate,
        return 1 - self.tp_rates


class Report:
    def __init__(self, criterion=None):
        self.criterion = criterion
        self.conf_matrix_train = []
        self.conf_matrix_test = []

    def append_fold(self, name, conf_matrix):
        if name == 'train':
            self.conf_matrix_train.append(conf_matrix)
        else:
            self.conf_matrix_test.append(conf_matrix)

    def __repr__(self):

        tp_rates = [m.tp_rates for m in self.conf_matrix_train]
        tn_rates = [m.tn_rates for m in self.conf_matrix_train]

        tp_rates = np.mean(np.array(tp_rates), axis=0)
        tn_rates = np.mean(np.array(tn_rates), axis=0)

        try:
            auc = sklearn.metrics.auc(1 - tn_rates, tp_rates)
        except:
            auc = -1

        try:
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(1 - tn_rates, tp_rates)(x), 0., 1.)
        except:
            eer = -1

        info = 'Area under curve (AUC): {:1.5f}\n'.format(auc) + \
               'Equal error rate (EER): {:1.5f}\n'.format(eer) + '\n'

        for i, criterion in enumerate(self.criterion):

            accuracy = [m.accuracy[i] for m in self.conf_matrix_test]
            precision = [m.precision[i] for m in self.conf_matrix_test]
            tp_rates = [m.tp_rates[i] for m in self.conf_matrix_test]
            tn_rates = [m.tn_rates[i] for m in self.conf_matrix_test]
            threshold = [m.threshold[i] for m in self.conf_matrix_test]

            info += criterion + '\n' \
                'Accuracy:  {:2.5f}+-{:2.5f}\n'.format(mean(accuracy), std(accuracy)) + \
                'Precision: {:2.5f}+-{:2.5f}\n'.format(mean(precision), std(precision)) + \
                'Sensitivity (TPR, 1-a type 1 error): {:2.5f}+-{:2.5f}\n'.format(mean(tp_rates), std(tp_rates)) + \
                'Specificity (TNR, 1-b type 2 error): {:2.5f}+-{:2.5f}\n'.format(mean(tn_rates), std(tn_rates)) + \
                'Threshold: {:2.5f}+-{:2.5f}\n'.format(mean(threshold), std(threshold)) + \
                '\n'

        return info


class Validation:
    def __init__(self, embeddings, labels, config, start_time=None):
        """
        :param embeddings:
        :param labels:
        """
        self.report_file = None
        self.embeddings = embeddings
        self.labels = labels
        assert (embeddings.shape[0] == len(labels))

        self.report = None
        self.config = config

        if self.config.metric == 0:
            upper_threshold = 4
        elif self.config.metric == 1:
            upper_threshold = np.pi
        else:
            raise ValueError('Undefined similarity metric {}'.format(self.config.metric))

        self.thresholds = np.linspace(0, upper_threshold, 100)

        self.elapsed_time = None
        self.start_time = start_time
        if self.start_time is None:
            self.start_time = time.monotonic()

    def evaluate(self):
        k_fold = KFold(n_splits=self.config.nrof_folds, shuffle=True, random_state=0)
        indices = np.arange(len(self.labels))

        criterion = ('Maximum accuracy criterion',
                     'False alarm rate target criterion (FAR = {})'.format(self.config.far_target))
        self.report = Report(criterion=criterion)

        calculator = SimilarityCalculator(self.embeddings, self.labels, metric=self.config.metric)

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            print('\rvalidation {}/{}'.format(fold_idx+1, self.config.nrof_folds), end=utils.end(fold_idx, self.config.nrof_folds))

            # evaluations with train set and define the best threshold for the fold
            calculator.set_indices(train_set)

            conf_matrix = ConfidenceMatrix(calculator, self.thresholds)
            self.report.append_fold('train', conf_matrix)

            # find the threshold that gives maximal accuracy
            accuracy_threshold = self.thresholds[np.argmax(conf_matrix.accuracy)]

            # find the threshold that gives FAR (FPR, 1-TNR) = far_target
            far_threshold = 0.0
            if np.max(conf_matrix.fp_rates) >= self.config.far_target:
                f = interpolate.interp1d(conf_matrix.fp_rates, self.thresholds, kind='slinear')
                far_threshold = f(self.config.far_target)

            # evaluations with test set
            calculator.set_indices(test_set)

            conf_matrix = ConfidenceMatrix(calculator, [accuracy_threshold, far_threshold])
            self.report.append_fold('test', conf_matrix)

        self.elapsed_time = time.monotonic() - self.start_time

    def write_report(self, path=None, dbase_info=None, emb_info=None):
        if self.config.file is None:
            dir_name = Path(path).expanduser()
            if dir_name.is_file():
                dir_name = dir_name.parent
            self.report_file = dir_name.joinpath('report.txt')
        else:
            self.report_file = Path(self.config.file).expanduser()

        with self.report_file.open('at') as f:
            f.write(''.join(['-'] * 64) + '\n')
            f.write('{} {}\n'.format(self.__class__.__name__, datetime.datetime.now()))
            f.write('elapsed time: {:.3f}\n'.format(time.monotonic() - self.start_time))
            f.write('git hash: {}\n'.format(utils.git_hash()))
            f.write('git diff: {}\n\n'.format(utils.git_diff()))
            f.write('{}\n'.format(dbase_info))
            f.write('{}\n'.format(emb_info))
            f.write('metric: {}\n\n'.format(self.config.metric))
            f.write(self.report.__repr__())

    def __repr__(self):
        """Representation of the database"""
        info = ('class {}\n'.format(self.__class__.__name__) +
                self.report.__repr__() +
                'Report has been written to the file: {}\n'.format(self.report_file) +
                'Elapsed time: {:.3f} sec'.format(self.elapsed_time))
        return info


class FalseExamples:
    def __init__(self, dbase, tfrecord, threshold, metric=0, subtract_mean=False):
        self.dbase = dbase
        self.embeddings = tfrecord.embeddings
        self.threshold = threshold
        self.metric = metric
        self.subtract_mean = subtract_mean

    def write_false_pairs(self, fpos_dir, fneg_dir, nrof_fpos_images=10, nrof_fneg_images=2):
        ioutils.makedirs(fpos_dir)
        ioutils.makedirs(fneg_dir)

        if self.subtract_mean:
            mean = np.mean(self.embeddings, axis=0)
        else:
            mean = 0

        for folder1 in range(self.dbase.nrof_folders):
            print('\rWrite false examples {}/{}'.format(folder1, self.dbase.nrof_folders),
                  end=utils.end(folder1, self.dbase.nrof_folders))

            files1, embeddings1 = self.dbase.extract_data(folder1, self.embeddings)

            # search false negative pairs
            similarities = pairwise_similarities(embeddings1 - mean, metric=self.metric)
            similarities = spatial.distance.squareform(similarities)

            for n in range(nrof_fpos_images):
                # find maximal distances
                i, k = np.unravel_index(np.argmax(similarities), similarities.shape)

                if similarities[i, k] > self.threshold:
                    self.write_image(similarities[i, k], files1[i], files1[k], fneg_dir)
                    similarities[[i, k], :] = -1
                    similarities[:, [i, k]] = -1
                else:
                    break

            # search false positive pairs
            for folder2 in range(folder1+1, self.dbase.nrof_folders):
                files2, embeddings2 = self.dbase.extract_data(folder2, self.embeddings)

                similarities = pairwise_similarities(embeddings1 - mean, embeddings2 - mean, metric=self.metric)

                for n in range(nrof_fneg_images):
                    # find minimal distances
                    i, k = np.unravel_index(np.argmin(similarities), similarities.shape)

                    if similarities[i, k] < self.threshold:
                        self.write_image(similarities[i, k], files1[i], files2[k], fpos_dir)
                        similarities[i, :] = np.Inf
                        similarities[:, k] = np.Inf
                    else:
                        break

    def generate_filename(self, dirname, distance, file1, file2):
        dir1 = os.path.basename(os.path.dirname(file1))
        name1 = os.path.splitext(os.path.basename(file1))[0]

        dir2 = os.path.basename(os.path.dirname(file2))
        name2 = os.path.splitext(os.path.basename(file2))[0]

        return os.path.join(dirname, '{:2.3f} & {}|{} & {}|{}.png'.format(distance, dir1, name1, dir2, name2))

    def generate_text(self, distance, file1, file2):

        def text(file):
            return os.path.join(os.path.basename(os.path.dirname(file)), os.path.splitext(os.path.basename(file))[0])

        return '{} & {}\n{:2.3f}/{:2.3f}'.format(text(file1), text(file2), distance, self.threshold)

    def write_image(self, distance, file1, file2, dirname, fsize=13):
        fname = self.generate_filename(dirname, distance, file1, file2)
        text = self.generate_text(distance, file1, file2)

        img1 = io.imread(file1)
        img2 = io.imread(file2)
        img = Image.fromarray(np.concatenate([img1, img2], axis=1))

        if sys.platform == 'win32':
            font = ImageFont.truetype("arial.ttf", fsize)
        else:
            font = ImageFont.truetype("LiberationSans-Regular.ttf", fsize)

        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, (0, 255, 0), font=font)

        img.save(fname)
