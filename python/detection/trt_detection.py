#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT Object detection.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import colorsys
import os
import random
import time

import cv2
import numpy as np
import tensorrt as trt

import common


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

WINDOW_NAME = "TensorRT detection example."


def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def normalize(img):
    img = np.asarray(img, dtype="float32")
    img = img / 127.5 - 1.0
    return img


def read_label_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


def random_colors(N):
    N = N + 1
    hsv = [(i / N, 1.0, 1.0) for i in range(N)]
    colors = list(
        map(lambda c: tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv)
    )
    random.shuffle(colors)
    return colors


def draw_rectangle(image, box, color, thickness=3):
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness)


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
        "--label", help="File path of label file.", required=True, type=str
    )
    parser.add_argument(
        "--videopath", help="File path of input video file.", default=None, type=str
    )
    parser.add_argument(
        "--output", help="File path of output vide file.", default=None, type=str
    )
    parser.add_argument("--width", help="Model input width", default=320, type=int)
    parser.add_argument("--height", help="Model input width", default=320, type=int)
    parser.add_argument(
        "--scoreThreshold", help="Score threshold.", default=0.5, type=float
    )
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 50)

    # Read label and generate random colors.
    labels = read_label_file(args.label) if args.label else None
    last_key = sorted(labels.keys())[len(labels.keys()) - 1]
    random.seed(42)
    colors = random_colors(last_key)

    # Video capture.
    if args.videopath == "":
        print("open camera.")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        print("open video file", args.videopath)
        cap = cv2.VideoCapture(args.videopath)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Input Video (height, width, fps): ", h, w, fps)

    model_name = os.path.splitext(os.path.basename(args.model))[0]

    # Load model.
    engine = get_engine(args.model)
    context = engine.create_execution_context()

    # Output Video file
    # Define the codec and create VideoWriter object
    video_writer = None
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    elapsed_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_im = cv2.resize(im, (args.width, args.height))
        normalized_im = normalize(resized_im)
        normalized_im = np.expand_dims(normalized_im, axis=0)

        # inference.
        start = time.perf_counter()
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = normalized_im
        outputs = common.do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        inference_time = (time.perf_counter() - start) * 1000

        boxs = outputs[1].reshape([int(outputs[0]), 4])
        for index, box in enumerate(boxs):
            if outputs[2][index] < args.scoreThreshold:
                continue

            # Draw bounding box.
            class_id = int(outputs[3][index])
            score = outputs[2][index]
            caption = "{0}({1:.2f})".format(labels[class_id - 1], score)

            xmin = int(box[0] * w)
            xmax = int(box[2] * w)
            ymin = int(box[1] * h)
            ymax = int(box[3] * h)
            draw_rectangle(frame, (xmin, ymin, xmax, ymax), colors[class_id])
            draw_caption(frame, (xmin, ymin - 10), caption)

        # Calc fps.
        elapsed_list.append(inference_time)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(inference_time)
        display_text = model_name + " " + fps_text + avg_text
        draw_caption(frame, (10, 30), display_text)

        # Output video file
        if video_writer is not None:
            video_writer.write(frame)

        # Display
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # When everything done, release the window
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
