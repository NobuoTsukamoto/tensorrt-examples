#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT Super resolution (ESRGAN).

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import math
import time

import cv2
import numpy as np
import tensorrt as trt

import common

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of onnx model.", required=True)
    parser.add_argument(
        "--input_shape",
        type=str,
        default="50,50",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--image", help="File path of input image.", type=str, required=True
    )
    parser.add_argument(
        "--output", help="File path of output image.", type=str, required=True
    )
    parser.add_argument("--upscale", help="upscale.", type=int, default=4)
    args = parser.parse_args()

    input_shape = tuple(map(int, args.input_shape.split(",")))

    # Read image file.
    im = cv2.imread(args.image)
    w, h, c = im.shape
    print("Input Image (height, width, channel): ", h, w, c)

    w_split = math.ceil(w / input_shape[0])
    h_split = math.ceil(h / input_shape[1])
    pad_width = w_split * input_shape[0] - w
    pad_height = h_split * input_shape[1] - h
    pad_im = cv2.copyMakeBorder(
        im, 0, pad_width, 0, pad_height, cv2.BORDER_CONSTANT, (0, 0, 0)
    )
    pad_im = cv2.cvtColor(pad_im, cv2.COLOR_BGR2RGB)
    pad_im = np.asarray(pad_im, dtype="float32")

    # Output image.
    output_image = np.zeros(
        (pad_im.shape[0] * args.upscale, pad_im.shape[1] * args.upscale, 3),
        dtype="uint8",
    )

    # Load model.
    engine = get_engine(args.model)
    context = engine.create_execution_context()

    # inference.
    elapsed_list = []
    for i in range(w_split):
        w_start = i * input_shape[0]
        w_end = (i + 1) * input_shape[0]

        for j in range(h_split):
            h_start = j * input_shape[1]
            h_end = (j + 1) * input_shape[1]

            input_im = pad_im[w_start:w_end, h_start:h_end, :]

            start = time.perf_counter()
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            inputs[0].host = np.array(input_im)
            outputs = common.do_inference_v2(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
            elapsed_list.append((time.perf_counter() - start) * 1000)

            # super resolution image.
            sr_image = np.array(outputs)
            sr_image = sr_image.clip(0.0, 255.0)
            sr_image = sr_image.reshape(
                input_shape[0] * args.upscale, input_shape[1] * args.upscale, 3
            )
            sr_image = np.asarray(sr_image, dtype="uint8")
            sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)

            output_image[
                w_start * args.upscale : w_end * args.upscale,
                h_start * args.upscale : h_end * args.upscale,
            ] = sr_image

    # Display fps
    fps_text = "Avg inference: {0:.2f}ms".format(np.mean(elapsed_list))
    print(fps_text)

    # Output image file
    cv2.imwrite(args.output, output_image)


if __name__ == "__main__":
    main()
