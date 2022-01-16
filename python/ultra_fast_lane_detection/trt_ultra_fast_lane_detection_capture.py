#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT Ultra-Fast-Lane-Detection example.

    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import os
import time

import cv2
import numpy as np
import scipy.special
import tensorrt as trt

import common

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

WINDOW_NAME = "TensorRT Ultra-Fast-Lane-Detection example."

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

MODEL_CONFIG = {
    "tusimple": {
        "input": (800, 288),
        "row_anchor": [
            64,  68,  72,  76,  80,  84,  88,  92,  96,
            100, 104, 108, 112, 116, 120, 124, 128, 132,
            136, 140, 144, 148, 152, 156, 160, 164, 168,
            172, 176, 180, 184, 188, 192, 196, 200, 204,
            208, 212, 216, 220, 224, 228, 232, 236, 240,
            244, 248, 252, 256, 260, 264, 268, 272, 276,
            280, 284
        ],
        "griding_num": 100,
        "num_per_lane": 56,
        "output": (101, 56, 4)
    },
    "culane": {
        "input": (800, 288),
        "row_anchor": [
            121, 131, 141, 150, 160, 170, 180, 189, 199,
            209, 219, 228, 238, 248, 258, 267, 277, 287
        ],
        "griding_num": 200,
        "num_per_lane": 18,
        "output": (201, 18, 4)
    },
}


def draw_circle(image, point):
    cv2.circle(image, point, 10, (246, 250, 250), -1)
    cv2.circle(image, point, 4, (255, 209, 0), 2)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of trt model.", required=True)
    parser.add_argument(
        "--model_config",
        type=str,
        default="tusimple",
        help='The name of the model. Either "tusimple" or "culane".',
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
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    engine = get_engine(args.model)
    context = engine.create_execution_context()
    config = MODEL_CONFIG[args.model_config]

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
        resized_im = cv2.resize(im, config["input"])
        normalized_im = normalize(resized_im)

        # inference.
        start = time.perf_counter()
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = np.ascontiguousarray(normalized_im)
        outputs = common.do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        inference_time = (time.perf_counter() - start) * 1000

        # post process.
        output = outputs[0]
        output = output.reshape(config["output"])
        output = output[:, ::-1, :]
        prob = scipy.special.softmax(output[:-1, :, :], axis=0)
        idx = np.arange(config["griding_num"]) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        output = np.argmax(output, axis=0)
        loc[output == config["griding_num"]] = 0
        output = loc

        col_sample = np.linspace(0, 800 - 1, config["griding_num"])
        col_sample_w = col_sample[1] - col_sample[0]

        lanes_points = []
        lanes_detected = []

        max_lanes = output.shape[1]
        for lane_num in range(max_lanes):
            lane_points = []

            # Check if there are any points detected in the lane.
            if np.sum(output[:, lane_num] != 0) > 2:
                lanes_detected.append(True)

                # Process each of the points for each lane.
                for point_num in range(output.shape[0]):
                    if output[point_num, lane_num] > 0:
                        x = int(output[point_num, lane_num] * col_sample_w * w / config["input"][0]) - 1
                        y = int(h * (config["row_anchor"][config["num_per_lane"] - 1 - point_num] / config["input"][1])) - 1
                        lane_point = [x, y]
                        lane_points.append(lane_point)
            else:
                lanes_detected.append(False)

            lanes_points.append(lane_points)

        # Draw lanes
        for lane_num, lane_points in enumerate(lanes_points):
            for lane_point in lane_points:
                draw_circle(frame, (lane_point[0], lane_point[1]))

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
