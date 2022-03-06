#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT Fast-SCNN example.

    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import os
import time

import cv2
import numpy as np
import tensorrt as trt

import common


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

WINDOW_NAME = "TensorRT Fast-SCNN example."

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def normalize(im):
    im = np.asarray(im, dtype="float32")
    im = (im / 255.0 - mean) / std
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


def create_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    ind = np.arange(256, dtype=np.uint8)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap


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
        default="576,768",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--input", help="File path of input image file.", required=True, type=str
    )
    parser.add_argument(
        "--output", help="File path of output image.", default=None, type=str
    )
    parser.add_argument(
        "--count", help="Nun of inference.", type=int, default=101
    )
    parser.add_argument(
        "--with_argmax", help="Model has argmax.", action="store_true"
    )
    args = parser.parse_args()

    # Initialize colormap
    colormap = create_label_colormap()

    # Load model.
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    input_shape = tuple(map(int, args.input_shape.split(",")))
    engine = get_engine(args.model)
    context = engine.create_execution_context()

    frame = cv2.imread(args.input)
    h, w, _ = frame.shape
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_im = cv2.resize(im, (input_shape[1], input_shape[0]))
    normalized_im = normalize(resized_im)

    elapsed_list = []

    for count in range(args.count):
        # inference.
        start = time.perf_counter()
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = np.ascontiguousarray(normalized_im)
        outputs = common.do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        seg_map = np.array(outputs[0])

        if args.with_argmax:
            seg_map = seg_map.reshape([input_shape[0], input_shape[1]])
        else:
            seg_map = seg_map.reshape([-1, input_shape[0], input_shape[1]])
            seg_map = np.argmax(seg_map, axis=0)
        inference_time = (time.perf_counter() - start) * 1000

        if count % 20 == 0:
            print("count: {0:04} Inference: {1:.2f} ms".format(count, inference_time))

        if (count != 0):
            elapsed_list.append(inference_time)

    if elapsed_list:
        print("Mean inference: {0:.2f} ms".format(np.mean(elapsed_list)))

    if args.output:
        # Post process.
        seg_map += 1
        seg_image = label_to_color_image(colormap, seg_map)
        seg_image = cv2.resize(seg_image, (w, h))
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) // 2 + seg_image // 2
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        draw_caption(im, (10, 30), model_name)

        # Output image file.
        cv2.imwrite(args.output, im)


if __name__ == "__main__":
    main()
