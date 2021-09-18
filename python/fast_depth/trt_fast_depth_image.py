#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT Fast depth (image).

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time
import os

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


def preprocess(im, input_shape):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, input_shape)
    im = np.asarray(im, dtype="float32")
    im = im.transpose(2, 0, 1)
    im = im / 255.0
    im = np.expand_dims(im, axis=0)
    return im


def postprocess(im, output_shape, is_apply_colormap=True):
    depth_im = im.reshape(output_shape[0], output_shape[1], 1)
    d_min = np.min(depth_im)
    d_max = np.max(depth_im)
    depth_im = (depth_im - d_min) / (d_max - d_min)
    depth_im = 255 * depth_im
    depth_im = np.asarray(depth_im, dtype="uint8")
    if is_apply_colormap:
        depth_im = cv2.applyColorMap(depth_im, cv2.COLORMAP_JET)
    return depth_im


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
    parser.add_argument("--model", help="File path of trt model.", required=True)
    parser.add_argument(
        "--input", help="File path of input video file.", required=True, type=str
    )
    parser.add_argument(
        "--output", help="File path of output image.", required=True, type=str
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="320,320",
        help="Specify an input shape for inference.",
    )
    args = parser.parse_args()

    # Load model.
    input_shape = tuple(map(int, args.input_shape.split(",")))
    engine = get_engine(args.model)
    context = engine.create_execution_context()
    model_name = os.path.splitext(os.path.basename(args.model))[0]

    # Read Image
    input_im = cv2.imread(args.input)
    h, w, c = input_im.shape
    print("Input image (height, width, channel): ", h, w, c)
    im = preprocess(input_im, input_shape)

    # inference.
    start = time.perf_counter()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    inputs[0].host = np.ascontiguousarray(im)
    outputs = common.do_inference_v2(
        context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
    )
    inference_time = (time.perf_counter() - start) * 1000

    # postprosess
    depth_im = postprocess(np.array(outputs[0]), input_shape)
    depth_im = cv2.resize(depth_im, (w, h))    
    im_v = cv2.vconcat([input_im, depth_im])

    # Display fps
    fps_text = "Inference: {0:.2f}ms".format(inference_time)
    display_text = model_name + " " + fps_text
    draw_caption(im_v, (10, 30), display_text)

    # Output image
    cv2.imwrite(args.output, im_v)


if __name__ == "__main__":
    main()
