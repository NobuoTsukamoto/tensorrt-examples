#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT Depth estimation (capture).

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

WINDOW_NAME = "TensorRT Depth estimation example."


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
        "--input_shape",
        type=str,
        default="640,480",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--output_shape",
        type=str,
        default="320,240",
        help="Specify an output shape for inference.",
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

    # Load model.
    input_shape = tuple(map(int, args.input_shape.split(",")))
    output_shape = tuple(map(int, args.output_shape.split(",")))[::-1]
    engine = get_engine(args.model)
    context = engine.create_execution_context()
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

        im = preprocess(frame, input_shape)

        # inference.
        start = time.perf_counter()
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = np.ascontiguousarray(im)
        outputs = common.do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        inference_time = (time.perf_counter() - start) * 1000

        # postprosess
        depth_im = postprocess(np.array(outputs[0]), output_shape)
        depth_im = cv2.resize(depth_im, (w, h))
        im_v = cv2.vconcat([frame, depth_im])

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
        draw_caption(im_v, (10, 30), display_text)

        # Output video file
        if video_writer is not None:
            video_writer.write(im_v)

        # Display
        cv2.imshow(WINDOW_NAME, im_v)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    # When everything done, release the window
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
