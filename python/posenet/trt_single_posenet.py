#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT PoseNet.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time

import cv2
import numpy as np
import tensorrt as trt

import common


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

WINDOW_NAME = "TensorRT PoseNet example."

NUM_KEYPOINTS = 17

KEYPOINT_EDGES = [
    # (0, 1),
    # (0, 2),
    # (1, 3),
    # (2, 4),
    # (0, 5),
    # (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def normalize(img):
    img = np.asarray(img, dtype="float32")
    img = img / 127.5 - 1.0
    return img


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4
    )
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
    )


def draw_circle(image, point):
    cv2.circle(image, point, 7, (246, 250, 250), -1)
    cv2.circle(image, point, 2, (255, 209, 0), 2)


def draw_line(image, point1, point2):
    cv2.line(image, point1, point2, (255, 209, 0), 5)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
        "--scoreThreshold", help="Score threshold.", default=0.5, type=float
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
    input_width = 257
    input_height = 353

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
        resized_im = cv2.resize(im, (input_width, input_height))
        normalized_im = normalize(resized_im)
        normalized_im = np.expand_dims(normalized_im, axis=0)

        # inference.
        start = time.perf_counter()
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = normalized_im
        outputs = common.do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        heatmaps = outputs[2].reshape((1, 23, 17, 17))
        heatmaps_shape = heatmaps.shape
        offset = outputs[0].reshape((1, 23, 17, 34))
        # forward_displacements = outputs[2]
        # backward_displacements = outputs[3]

        height = heatmaps_shape[1]
        width = heatmaps_shape[2]
        num_keypoints = heatmaps_shape[3]

        keypoint_positions = []
        for keypoint in range(num_keypoints):
            max_val = heatmaps[0][0][0][keypoint]
            max_row = 0
            max_col = 0
            for row in range(height):
                for col in range(width):
                    if heatmaps[0][row][col][keypoint] > max_val:
                        max_val = heatmaps[0][row][col][keypoint]
                        max_row = row
                        max_col = col

            keypoint_positions.append((max_row, max_col))

        y_coords = []
        x_coords = []
        confidence_scores = []
        for i, position in enumerate(keypoint_positions):
            position_y = position[0]
            position_x = position[1]

            y_coords.append(
                int(
                    position_y / float(height - 1.0) * h
                    + offset[0][position_y][position_x][i]
                )
            )
            x_coords.append(
                int(
                    position_x / float(width - 1.0) * w
                    + offset[0][position_y][position_x][i + num_keypoints]
                )
            )
            confidence_scores.append(sigmoid(heatmaps[0][position_y][position_x][i]))

        inference_time = (time.perf_counter() - start) * 1000

        for i in range(NUM_KEYPOINTS):
            # if confidence_scores[i] < args.scoreThreshold:
            #     continue
            draw_circle(frame, (x_coords[i], y_coords[i]))

        for keypoint_start, keypoint_end in KEYPOINT_EDGES:
            if (
                keypoint_start < 0
                or keypoint_start >= NUM_KEYPOINTS
                or keypoint_end < 0
                or keypoint_end >= NUM_KEYPOINTS
            ):
                continue

            if (
                confidence_scores[keypoint_start] < args.scoreThreshold
                or confidence_scores[keypoint_end] < args.scoreThreshold
            ):
                continue

            draw_line(
                frame,
                (x_coords[keypoint_start], y_coords[keypoint_start]),
                (x_coords[keypoint_end], y_coords[keypoint_end]),
            )

        # Calc fps.
        elapsed_list.append(inference_time)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms, FPS: {1:.2f}".format(
                avg_elapsed_ms, 1000 / avg_elapsed_ms
            )

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(inference_time)
        display_text = fps_text + avg_text
        draw_caption(frame, (10, 50), display_text)

        # Output video file
        if video_writer is not None:
            video_writer.write(frame)

        # Display
        frame = cv2.resize(frame, (int(w / 2.5), int(h / 2.5)))
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
