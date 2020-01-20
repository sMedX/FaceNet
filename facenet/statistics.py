
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys

import os
import datetime
import numpy as np
from skimage import io
import sklearn
from sklearn.model_selection import KFold
from scipy import spatial, interpolate
from scipy.optimize import brentq
import math
import pathlib

from facenet import utils


def pairwise_distances(xa, xb=None, metric=0):
    if metric == 0:
        # squared Euclidian distance
        if xb is None:
            dist = spatial.distance.pdist(xa, metric='sqeuclidean')
        else:
            dist = spatial.distance.cdist(xa, xb, metric='sqeuclidean')
    elif metric == 1:
        # distance based on cosine similarity
        if xb is None:
            dist = spatial.distance.pdist(xa, metric='cosine')
        else:
            dist = spatial.distance.cdist(xa, xb, metric='cosine')
        dist = np.arccos(1 - dist) / math.pi
    else:
        raise 'Undefined distance metric %d' % metric

    return dist


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


def compute_distances(embeddings, labels, metric=0):
    embeddings = split_embeddings(embeddings, labels)
    distances = []

    for i, emb1 in enumerate(embeddings):
        distances.append([])
        for k, emb2 in enumerate(embeddings[:i]):
            distances[i].append(pairwise_distances(emb1, emb2, metric=metric))
        distances[i].append(pairwise_distances(emb1, metric=metric))

    return distances


class ConfidenceMatrix:
    def __init__(self, distances, threshold):

        self.threshold = np.array(threshold, ndmin=1)

        self.tp = np.zeros(self.threshold.size)
        self.tn = np.zeros(self.threshold.size)
        self.fp = np.zeros(self.threshold.size)
        self.fn = np.zeros(self.threshold.size)

        for i, distances_i in enumerate(distances):
            for k, distances_k in enumerate(distances_i):
                for n, threshold in enumerate(self.threshold):
                    count = np.count_nonzero(distances_k < threshold)

                    if i == k:
                        self.tp[n] += count
                        self.fn[n] += distances_k.size - count
                    else:
                        self.fp[n] += count
                        self.tn[n] += distances_k.size - count

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
    def __init__(self, criterion=None, nrof_folds=5):
        self.criterion = criterion
        self.nrof_folds = nrof_folds
        self.conf_matrix_train = []
        self.conf_matrix_test = []

    def append_fold(self, name, conf_matrix):
        if name == 'train':
            self.conf_matrix_train.append(conf_matrix)
        else:
            self.conf_matrix_test.append(conf_matrix)

    def __repr__(self):
        auc = -1
        eer = -1

        if len(self.conf_matrix_train) > 0:
            tp_rates = [m.tp_rates for m in self.conf_matrix_train]
            tn_rates = [m.tn_rates for m in self.conf_matrix_train]

            tp_rates = np.mean(np.array(tp_rates), axis=0)
            tn_rates = np.mean(np.array(tn_rates), axis=0)

            try:
                auc = sklearn.metrics.auc(1 - tn_rates, tp_rates)
            except Exception:
                pass

            try:
                eer = brentq(lambda x: 1. - x - interpolate.interp1d(1 - tn_rates, tp_rates)(x), 0., 1.)
            except Exception:
                pass

        accuracy = [m.accuracy for m in self.conf_matrix_test]
        precision = [m.precision for m in self.conf_matrix_test]
        tp_rates = [m.tp_rates for m in self.conf_matrix_test]
        tn_rates = [m.tn_rates for m in self.conf_matrix_test]
        threshold = [m.threshold for m in self.conf_matrix_test]

        info = self.criterion + '\n' + \
            'Accuracy:  {:2.5f}+-{:2.5f}\n'.format(mean(accuracy), std(accuracy)) + \
            'Precision: {:2.5f}+-{:2.5f}\n'.format(mean(precision), std(precision)) + \
            'Sensitivity (TPR, 1-a type 1 error): {:2.5f}+-{:2.5f}\n'.format(mean(tp_rates), std(tp_rates)) + \
            'Specificity (TNR, 1-b type 2 error): {:2.5f}+-{:2.5f}\n'.format(mean(tn_rates), std(tn_rates)) + '\n'

        if len(self.conf_matrix_train) > 0:
            info += 'Area under curve (AUC): {:1.5f}\n'.format(auc) + \
                    'Equal error rate (EER): {:1.5f}\n'.format(eer) + '\n'

        info += 'Threshold: {:2.5f}+-{:2.5f}\n'.format(mean(threshold), std(threshold))
        return info


class Validation:
    def __init__(self, thresholds, embeddings, labels,
                 far_target=1e-3, nrof_folds=10,
                 metric=0, subtract_mean=False):
        """

        :param thresholds:
        :param embeddings:
        :param labels:
        :param far_target: target false alarm rate (face pairs that was incorrectly classified as the same)
        :param nrof_folds:
        :param metric:
        :param subtract_mean:
        """

        self.subtract_mean = subtract_mean
        self.metric = metric
        self.far_target = far_target

        self.labels = np.array(labels)
        self.embeddings = embeddings
        assert (embeddings.shape[0] == len(labels))

        k_fold = KFold(n_splits=nrof_folds, shuffle=False)
        indices = np.arange(len(labels))

        self.report_acc = Report(criterion='Maximum accuracy criterion', nrof_folds=nrof_folds)
        self.report_far = Report(criterion='False alarm rate target criterion', nrof_folds=nrof_folds)

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            print('\rvalidation {}/{}'.format(fold_idx, nrof_folds), end=utils.end(fold_idx, nrof_folds))

            # evaluations with train set and define the best threshold for the fold
            distances = compute_distances(self.embeddings[train_set], self.labels[train_set], metric=0)
            conf_matrix = ConfidenceMatrix(distances, thresholds)

            self.report_acc.append_fold('train', conf_matrix)

            # find the threshold that gives maximal accuracy
            accuracy_threshold = thresholds[np.argmax(conf_matrix.accuracy)]

            # find the threshold that gives FAR (FPR, 1-TNR) = far_target
            far_threshold = 0.0
            if np.max(conf_matrix.fp_rates) >= self.far_target:
                f = interpolate.interp1d(conf_matrix.fp_rates, thresholds, kind='slinear')
                far_threshold = f(self.far_target)

            # evaluations with test set
            distances = compute_distances(self.embeddings[test_set], self.labels[test_set], metric=0)

            self.report_acc.append_fold('test', ConfidenceMatrix(distances, accuracy_threshold))
            self.report_far.append_fold('test', ConfidenceMatrix(distances, far_threshold))

        print(self.report_acc)
        print(self.report_far)

    def write_report(self, elapsed_time, args, file=None, dbase_info=None):
        if file is None:
            dir_name = pathlib.Path(args.model).expanduser()
            if dir_name.is_file():
                dir_name = dir_name.parent
            file = dir_name.joinpath('report.txt')
        else:
            file = pathlib.Path(file).expanduser()

        git_hash, git_diff = utils.git_hash()
        with file.open('at') as f:
            f.write('{}\n'.format(datetime.datetime.now()))
            f.write('git hash: {}\n'.format(git_hash))
            f.write('git diff: {}\n'.format(git_diff))
            f.write('{}'.format(dbase_info))
            f.write('model: {}\n'.format(args.model))
            f.write('embedding size: {}\n'. format(self.embeddings.shape[1]))
            f.write('elapsed time: {}\n'.format(elapsed_time))
            f.write('time per image: {}\n'.format(elapsed_time/self.embeddings.shape[0]))
            f.write('distance metric: {}\n'.format(self.metric))
            f.write('subtract mean: {}\n'.format(self.subtract_mean))
            f.write('\n')
            f.write(self.report_acc.__repr__())
            f.write('\n')
            f.write(self.report_far.__repr__())
            f.write('\n')

        print('Report has been printed to the file: {}'.format(file))


class FalseExamples:
    def __init__(self, dbase, tfrecord, threshold, metric=0, subtract_mean=False):
        self.dbase = dbase
        self.embeddings = tfrecord.embeddings
        self.threshold = threshold
        self.metric = metric
        self.subtract_mean = subtract_mean

    def write_false_pairs(self, fpos_dir, fneg_dir, nrof_fpos_images=10, nrof_fneg_images=2):

        if not os.path.isdir(fpos_dir):
            os.makedirs(fpos_dir)

        if not os.path.isdir(fneg_dir):
            os.makedirs(fneg_dir)

        if self.subtract_mean:
            mean = np.mean(self.embeddings, axis=0)
        else:
            mean = 0

        for folder1 in range(self.dbase.nrof_folders):
            print('\rWrite false examples {}/{}'.format(folder1, self.dbase.nrof_folders),
                  end=utils.end(folder1, self.dbase.nrof_folders))

            files1, embeddings1 = self.dbase.extract_data(folder1, self.embeddings)

            # search false negative pairs
            distances = pairwise_distances(embeddings1 - mean, metric=self.metric)
            distances = spatial.distance.squareform(distances)

            for n in range(nrof_fpos_images):
                # find maximal distances
                i, k = np.unravel_index(np.argmax(distances), distances.shape)

                if distances[i, k] > self.threshold:
                    self.write_image(distances[i, k], files1[i], files1[k], fneg_dir)
                    distances[[i, k], :] = -1
                    distances[:, [i, k]] = -1
                else:
                    break

            # search false positive pairs
            for folder2 in range(folder1+1, self.dbase.nrof_folders):
                files2, embeddings2 = self.dbase.extract_data(folder2, self.embeddings)

                distances = pairwise_distances(embeddings1-mean, embeddings2-mean, metric=self.metric)

                for n in range(nrof_fneg_images):
                    # find minimal distances
                    i, k = np.unravel_index(np.argmin(distances), distances.shape)

                    if distances[i, k] < self.threshold:
                        self.write_image(distances[i, k], files1[i], files2[k], fpos_dir)
                        distances[i, :] = np.Inf
                        distances[:, k] = np.Inf
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
