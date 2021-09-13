#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT Face landmark.

    Copyright (c) 2021 Nobuo Tsukamoto
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
        "--videopath", help="File path of input video file.", default=None, type=str
    )
    parser.add_argument(
        "--output", help="File path of output vide file.", default=None, type=str
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="160,160",
        help="Specify an input shape for inference (w, h).",
    )
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 50)

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

    # Load model.
    engine = get_engine(args.model)
    context = engine.create_execution_context()
    input_shape = tuple(map(int, args.input_shape.split(",")))
    model_name = os.path.splitext(os.path.basename(args.model))[0]

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
        # frame = np.ones([h, w, 3], dtype=np.uint8)
        landmarks = np.array(outputs[0]).reshape([-1, 2])
        for landmark in landmarks:
            center = (int(landmark[0] * w), int(landmark[1] * h))
            draw_circle(frame, center)

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
