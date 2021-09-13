#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT YuNet .

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.

"""
# This source corresponds to TensorRT by referring to the following.
# https://github.com/opencv/opencv_zoo/blob/e6e1754dcf0c058cad20166498e68f17c71fa3b1/models/face_detection_yunet/yunet.py

from itertools import product

import numpy as np
import cv2 as cv
import tensorrt as trt

import common


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


class TrtYuNet:
    def __init__(
        self,
        model_path,
        input_size=[160, 120],
        conf_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
        keep_top_k=750,
    ):
        self._model_path = model_path
        self._engine = self._getEngine(self._model_path)
        self._context = self._engine.create_execution_context()

        self._input_size = input_size  # [w, h]
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._keep_top_k = keep_top_k

        self._min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self._steps = [8, 16, 32, 64]
        self._variance = [0.1, 0.2]

        # Generate priors
        self._priorGen()

    @property
    def name(self):
        return self.__class__.__name__

    def setInputSize(self, input_size):
        self._input_size = input_size  # [w, h]

        # Regenerate priors
        self._priorGen()

    def _getEngine(self, engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _preprocess(self, image):
        image = cv.resize(image, (self._input_size[0], self._input_size[1]))
        image = np.asarray(image, dtype="float32")
        image = image.transpose(2, 0, 1)
        return image

    def infer(self, image):
        # Preprocess
        input_blob = self._preprocess(image)

        # Forward
        inputs, outputs, bindings, stream = common.allocate_buffers(self._engine)
        inputs[0].host = np.ascontiguousarray(input_blob)
        outputs = common.do_inference_v2(
            self._context,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )

        # Postprocess
        results = self._postprocess(outputs)

        return results

    def _postprocess(self, output_blob):
        # Decode
        dets = self._decode(output_blob)

        # NMS
        keepIdx = cv.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self._conf_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
        )  # box_num x class_num
        if len(keepIdx) > 0:
            dets = dets[keepIdx]
            dets = np.squeeze(dets, axis=1)
            return dets[: self._keep_top_k]
        else:
            return np.empty(shape=(0, 15))

    def _priorGen(self):
        w, h = self._input_size
        feature_map_2th = [int(int((h + 1) / 2) / 2), int(int((w + 1) / 2) / 2)]
        feature_map_3th = [int(feature_map_2th[0] / 2), int(feature_map_2th[1] / 2)]
        feature_map_4th = [int(feature_map_3th[0] / 2), int(feature_map_3th[1] / 2)]
        feature_map_5th = [int(feature_map_4th[0] / 2), int(feature_map_4th[1] / 2)]
        feature_map_6th = [int(feature_map_5th[0] / 2), int(feature_map_5th[1] / 2)]

        feature_maps = [
            feature_map_3th,
            feature_map_4th,
            feature_map_5th,
            feature_map_6th,
        ]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self._min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):  # i->h, j->w
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h

                    cx = (j + 0.5) * self._steps[k] / w
                    cy = (i + 0.5) * self._steps[k] / h

                    priors.append([cx, cy, s_kx, s_ky])
        self.priors = np.array(priors, dtype=np.float32)

    def _decode(self, outputBlob):
        loc = np.array(outputBlob[0]).reshape([-1, 14])
        conf = np.array(outputBlob[1]).reshape([-1, 2])
        iou = np.array(outputBlob[2]).reshape([-1, 1])

        # get score
        cls_scores = conf[:, 1]
        iou_scores = iou[:, 0]
        # clamp
        _idx = np.where(iou_scores < 0.0)
        iou_scores[_idx] = 0.0
        _idx = np.where(iou_scores > 1.0)
        iou_scores[_idx] = 1.0
        scores = np.sqrt(cls_scores * iou_scores)
        scores = scores[:, np.newaxis]

        scale = np.array(self._input_size)

        # get bboxes
        bboxes = np.hstack(
            (
                (
                    self.priors[:, 0:2]
                    + loc[:, 0:2] * self._variance[0] * self.priors[:, 2:4]
                )
                * scale,
                (self.priors[:, 2:4] * np.exp(loc[:, 2:4] * self._variance)) * scale,
            )
        )
        # (x_c, y_c, w, h) -> (x1, y1, w, h)
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

        # get landmarks
        landmarks = np.hstack(
            (
                (
                    self.priors[:, 0:2]
                    + loc[:, 4:6] * self._variance[0] * self.priors[:, 2:4]
                )
                * scale,
                (
                    self.priors[:, 0:2]
                    + loc[:, 6:8] * self._variance[0] * self.priors[:, 2:4]
                )
                * scale,
                (
                    self.priors[:, 0:2]
                    + loc[:, 8:10] * self._variance[0] * self.priors[:, 2:4]
                )
                * scale,
                (
                    self.priors[:, 0:2]
                    + loc[:, 10:12] * self._variance[0] * self.priors[:, 2:4]
                )
                * scale,
                (
                    self.priors[:, 0:2]
                    + loc[:, 12:14] * self._variance[0] * self.priors[:, 2:4]
                )
                * scale,
            )
        )

        dets = np.hstack((bboxes, landmarks, scores))
        return dets
