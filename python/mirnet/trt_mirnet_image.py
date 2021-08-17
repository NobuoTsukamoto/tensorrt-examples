#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT MIRNet (Image input).

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

WINDOW_NAME = "TensorRT MIRNet example."


def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def normalize(im):
    im = np.asarray(im, dtype="float32")
    im = im / 255.0
    im = np.expand_dims(im, axis=0)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of trt model.", required=True)
    parser.add_argument(
        "--input_shape",
        type=str,
        default="320,320",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--image", help="File path of input image.", type=str, required=True
    )
    parser.add_argument(
        "--output", help="File path of output image.", type=str, required=True
    )
    args = parser.parse_args()

    # Read image file.
    im = cv2.imread(args.image)
    w, h, c = im.shape

    print("Input Image (height, width, channel): ", h, w, c)

    # Load model.
    input_shape = tuple(map(int, args.input_shape.split(",")))
    engine = get_engine(args.model)
    context = engine.create_execution_context()

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    resized_im = cv2.resize(im, input_shape)
    normalized_im = normalize(resized_im)

    # inference.
    start = time.perf_counter()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    inputs[0].host = normalized_im
    outputs = common.do_inference_v2(
        context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
    )

    inference_time = (time.perf_counter() - start) * 1000

    output_image = np.array(outputs) * 255.0
    output_image = output_image.clip(0.0, 255.0)
    output_image = output_image.reshape(input_shape[0], input_shape[1], 3)
    output_image = np.asarray(output_image, dtype="uint8")
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Display fps
    fps_text = "Inference: {0:.2f}ms".format(inference_time)
    print(fps_text)

    # Output image file
    cv2.imwrite(args.output, output_image)


if __name__ == "__main__":
    main()
