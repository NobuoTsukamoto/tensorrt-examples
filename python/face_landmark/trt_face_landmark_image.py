#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT Face landmark image.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time

import cv2
import numpy as np
import tensorrt as trt

import common


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

WINDOW_NAME = "TensorRT Face landmark example."


def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def draw_rectangle(image, box, color, thickness=3):
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness)


def draw_circle(image, point):
    cv2.circle(image, point, 7, (246, 250, 250), -1)
    cv2.circle(image, point, 2, (255, 209, 0), 2)


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of onnx model.", required=True)
    parser.add_argument(
        "--input", help="File path of input video file.", required=True, type=str
    )
    parser.add_argument(
        "--output", help="File path of output image.", required=True, type=str
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="160,160",
        help="Specify an input shape for inference (w, h).",
    )
    args = parser.parse_args()

    # Load model.
    engine = get_engine(args.model)
    context = engine.create_execution_context()
    input_shape = tuple(map(int, args.input_shape.split(",")))

    # Read Image
    input_im = cv2.imread(args.input)
    h, w, c = input_im.shape
    print("Input image (height, width, channel): ", h, w, c)
    im = cv2.cvtColor(input_im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, input_shape)
    im = np.expand_dims(im, axis=0)
    im = np.asarray(im, dtype="float32")

    # inference.
    start = time.perf_counter()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    inputs[0].host = im
    outputs = common.do_inference_v2(
        context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
    )
    inference_time = (time.perf_counter() - start) * 1000

    # Display result.
    landmarks = np.array(outputs[0]).reshape([-1, 2])
    for landmark in landmarks:
        center = (int(landmark[0] * w), int(landmark[1] * h))
        draw_circle(input_im, center)

    # Display fps
    fps_text = "Inference: {0:.2f}ms".format(inference_time)
    draw_caption(input_im, (10, 30), fps_text)

    # Output image
    cv2.imwrite(args.output, input_im)


if __name__ == "__main__":
    main()
