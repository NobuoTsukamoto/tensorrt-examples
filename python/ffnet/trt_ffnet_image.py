#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT FFNet example.

    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import os

import cv2
import numpy as np
import tensorrt as trt

import common


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

WINDOW_NAME = "TensorRT FFNet example."

mean = [127.5, 127.5, 127.5]
std = [127.5, 127.5, 127.5]

colormap = np.array(
    (
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
        (0, 0, 0),
    ),
    np.uint8,
)


def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def normalize(im):
    im = im.astype(np.float32)
    im = (im - mean) / std
    im = im.transpose(2, 0, 1)
    im = np.expand_dims(im, axis=0)
    return im.astype("float32")


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
    )


def label_to_color_image(colormap, label):
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of trt model.", required=True)
    parser.add_argument(
        "--input_shape",
        type=str,
        default="1024,2048",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--output_shape",
        type=str,
        default="1024,2048",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--input", help="File path of input image file.", required=True, type=str
    )
    parser.add_argument(
        "--output", help="File path of output image.", required=True, type=str
    )
    args = parser.parse_args()

    # Load model.
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    input_shape = tuple(map(int, args.input_shape.split(",")))
    output_shape = tuple(map(int, args.output_shape.split(",")))
    engine = get_engine(args.model)
    context = engine.create_execution_context()

    frame = cv2.imread(args.input)
    h, w, _ = frame.shape
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_im = cv2.resize(im, (input_shape[1], input_shape[0]))
    normalized_im = normalize(resized_im)

    # inference.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    inputs[0].host = np.ascontiguousarray(normalized_im)
    outputs = common.do_inference_v2(
        context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
    )

    seg_map = np.array(outputs[0])
    seg_map = seg_map.reshape([output_shape[0], output_shape[1]])

    # Post process.
    seg_image = label_to_color_image(colormap, seg_map)
    seg_image = cv2.resize(seg_image, (w, h))
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) // 2 + seg_image // 2
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    draw_caption(im, (10, 30), model_name)

    # Output image file.
    cv2.imwrite(args.output, im)


if __name__ == "__main__":
    main()
