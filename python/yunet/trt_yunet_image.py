#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT YuNet demo.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time
import os

import cv2
import numpy as np

from trt_yunet import TrtYuNet


def draw_caption(image, point, caption):
    cv2.putText(
        image,
        caption,
        (point[0], point[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        image,
        caption,
        (point[0], point[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        1,
    )


def draw_rectangle(image, bbox, color=(255, 209, 0), thickness=3):
    cv2.rectangle(
        image,
        (bbox[0], bbox[1]),
        (bbox[0] + bbox[2], bbox[1] + bbox[3]),
        color,
        thickness,
    )


def draw_circle(image, point):
    cv2.circle(image, point, 7, (246, 250, 250), -1)
    cv2.circle(image, point, 2, (255, 209, 0), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of trt model.", required=True)
    parser.add_argument(
        "--input_shape",
        type=str,
        default="160,120",
        help="Specify an input shape for inference (w, h).",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.9,
        help="Filter out faces of confidence < conf_threshold.",
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.3,
        help="Suppress bounding boxes of iou >= nms_threshold.",
    )
    parser.add_argument(
        "--top_k", type=int, default=5000, help="Keep top_k bounding boxes before NMS."
    )
    parser.add_argument(
        "--keep_top_k",
        type=int,
        default=750,
        help="Keep keep_top_k bounding boxes after NMS.",
    )
    parser.add_argument(
        "--input", help="File path of input video file.", required=True, type=str
    )
    parser.add_argument(
        "--output", help="File path of output image.", required=True, type=str
    )
    args = parser.parse_args()

    # Read Image
    im = cv2.imread(args.input)
    h, w, c = im.shape

    print("Input image (height, width, channel): ", h, w, c)

    # Yunet model.
    input_shape = tuple(map(int, args.input_shape.split(",")))
    model = TrtYuNet(
        model_path=args.model,
        input_size=input_shape,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        top_k=args.top_k,
        keep_top_k=args.keep_top_k,
    )
    w_scale = w / input_shape[0]
    h_scale = h / input_shape[1]

    # inference.
    start = time.perf_counter()
    results = model.infer(im)
    inference_time = (time.perf_counter() - start) * 1000

    # Draw result.
    for det in results:
        xmin = int(det[0] * w_scale)
        xmax = int(det[2] * w_scale)
        ymin = int(det[1] * h_scale)
        ymax = int(det[3] * h_scale)

        draw_rectangle(im, (xmin, ymin, xmax, ymax))

        conf = det[-1]
        draw_caption(im, (xmin, ymin - 10), "{:.4f}".format(conf))

        landmarks = det[4:14].astype(np.int32).reshape((5, 2))
        for landmark in landmarks:
            draw_circle(im, (int(landmark[0] * w_scale), int(landmark[1] * h_scale)))

    # Display inference time.
    fps_text = "Inference: {0:.2f}ms".format(inference_time)
    display_text = fps_text
    draw_caption(im, (10, 30), display_text)

    # Output image
    cv2.imwrite(args.output, im)


if __name__ == "__main__":
    main()
