#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT U^2-Net.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time
import os
from enum import Enum

import cv2
import numpy as np
import tensorrt as trt

import common


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

WINDOW_NAME = "TensorRT U^2-Net example."


class Mode(Enum):
    OUTPUT = 1
    MASK = 2
    MASK_INV = 3


def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def normalize(im):
    im = np.asarray(im, dtype="float32")
    im = im / 127.5 - 1.0
    im = im.transpose(2, 0, 1)
    im = np.expand_dims(im, axis=0)
    return im


def get_next_mode(current_mode):
    current_mode_idx = list(Mode).index(current_mode)
    next_mode_idx = (current_mode_idx + 1) % len(Mode)
    next_mode = list(Mode)[next_mode_idx]
    return next_mode


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
        "--input_shape",
        type=str,
        default="320,320",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--videopath", help="File path of input video file.", default=None, type=str
    )
    parser.add_argument("--output", help="File path of output image.", type=str)
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

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    mode = Mode.OUTPUT

    # Load model.
    input_shape = tuple(map(int, args.input_shape.split(",")))
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
        resized_im = cv2.resize(im, input_shape)
        normalized_im = normalize(resized_im)

        # inference.
        start = time.perf_counter()
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = np.ascontiguousarray(normalized_im)
        outputs = common.do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        inference_time = (time.perf_counter() - start) * 1000

        output_im = (np.array(outputs[0]) + 1) * 127.5
        output_im = output_im.clip(0.0, 255.0)
        output_im = output_im.reshape(input_shape[0], input_shape[1], 1)
        output_im = np.asarray(output_im, dtype="uint8")
        if mode == Mode.MASK:
            _, output_im = cv2.threshold(output_im, 127, 255, cv2.THRESH_BINARY)
            output_im = cv2.cvtColor(output_im, cv2.COLOR_GRAY2BGR)
            output_im = cv2.resize(output_im, (w, h))
            masked_im = cv2.bitwise_or(frame, output_im)

        elif mode == Mode.MASK_INV:
            _, output_im = cv2.threshold(output_im, 127, 255, cv2.THRESH_BINARY)
            output_im = cv2.cvtColor(output_im, cv2.COLOR_GRAY2BGR)
            output_im = cv2.resize(output_im, (w, h))
            masked_im = cv2.bitwise_or(frame, output_im)

        else:
            # _, output_im = cv2.threshold(output_im, 127, 255, cv2.THRESH_BINARY)
            output_im = cv2.resize(output_im, (w, h))
            masked_im = cv2.cvtColor(output_im, cv2.COLOR_GRAY2BGR)

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
        draw_caption(masked_im, (10, 30), display_text)

        # Output video file
        if video_writer is not None:
            video_writer.write(masked_im)

        # Display
        masked_im = cv2.resize(masked_im, (w // 2, h // 2))
        cv2.imshow(WINDOW_NAME, masked_im)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

        elif key == ord(" "):
            mode = get_next_mode(mode)
            print("Changed mode: ", mode)

    # When everything done, release the window
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
