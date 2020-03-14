# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import numpy as np
from PIL import Image
from facenet import ioutils


def image_processing(image, box, size, margin=0):
    if not isinstance(image, Image.Image):
        raise ValueError('Input must be PIL.Image')

    if isinstance(size, int):
        size = (size, size)

    height_margin = round(box.height*margin)
    width_margin = round(box.width*margin)

    cropped = image.crop((box.left - width_margin, box.top - height_margin,
                          box.right + width_margin, box.bottom + height_margin))

    resized = cropped.resize(size, Image.ANTIALIAS)

    return resized


class BoundingBox:
    def __init__(self, left, top, width, height, confidence=None):
        self.left = int(np.round(left))
        self.right = int(np.round(left + width)) + 1

        self.top = int(np.round(top))
        self.bottom = int(np.round(top + height)) + 1

        self.width = self.right - self.left - 1
        self.height = self.bottom - self.top - 1
        self.confidence = confidence

    def info(self, mode=False):
        if mode is False:
            return '{}'.format([self.left, self.top, self.width, self.height, self.confidence])
        info = "left = {}, top = {}, width = {}, height = {}, confidence = {}".format(self.left, self.top, self.width, self.height, self.confidence)
        return info

    def __repr__(self):
        return self.info(mode=True)

    @property
    def left_upper(self):
        return self.left, self.top

    @property
    def right_lower(self):
        return self.right, self.bottom

    @property
    def confidence_as_string(self):
        return str(np.round(self.confidence, 3))


class MTCNN:
    def __init__(self):
        from mtcnn.mtcnn import MTCNN
        self.__detector = MTCNN().detect_faces
        self.__mode = 'BGR'

    def detector(self, image):
        faces = self.__detector(ioutils.pil2array(image, self.__mode))
        bboxes = []

        for face in faces:
            box = face['box']
            bbox = BoundingBox(left=box[0], top=box[1], width=box[2], height=box[3], confidence=face['confidence'])
            bboxes.append(bbox)

        return bboxes


class FasterRCNNv3:
    def __init__(self, gpu_memory_fraction=1.0):
        from .frcnnv3 import detector
        self.__detector = detector.FaceDetector(gpu_memory_fraction=gpu_memory_fraction).get_faces
        self.__mode = 'RGB'

    def detector(self, image):

        boxes, scores = self.__detector(ioutils.pil2array(image, self.__mode))
        bboxes = []

        for (y1, x1, y2, x2), score in zip(boxes, scores):
            bbox = BoundingBox(left=x1, top=y1, width=x2-x1, height=y2-y1, confidence=score)
            bboxes.append(bbox)

        return bboxes


class FaceDetector:
    def __init__(self, detector='frcnnv3', gpu_memory_fraction=1.0):
        self.detector = detector

        if self.detector == 'pypimtcnn':
            self.__detector = MTCNN().detector

        elif self.detector == 'frcnnv3':
            self.__detector = FasterRCNNv3(gpu_memory_fraction=gpu_memory_fraction).detector

        else:
            raise 'Undefined face detector type {}'.format(self.detector)

    def detect(self, image):
        return self.__detector(image)

    def __repr__(self):
        return 'Face detector {} has been created'.format(self.detector)

