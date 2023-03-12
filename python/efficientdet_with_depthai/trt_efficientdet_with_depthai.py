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
import math

import cv2
import depthai as dai
import numpy as np
from calc import HostSpatialsCalc

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
    # img = img / 127.5 - 1.0
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
        "--input_shape",
        type=str,
        default="512,512",
        help="Specify an input shape for inference.",
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

    # Read label and generate random colors.
    labels = read_label_file(args.label) if args.label else None
    last_key = sorted(labels.keys())[len(labels.keys()) - 1]
    random.seed(42)
    colors = random_colors(last_key)

    # Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p.
    # Otherwise (False), the aligned depth is automatically upscaled to 1080p
    downscaleColor = True
    fps = 30
    # The disparity is computed at this resolution, then upscaled to RGB resolution
    monoResolution = dai.MonoCameraProperties.SensorResolution.THE_480_P

    # Create pipeline
    pipeline = dai.Pipeline()
    device = dai.Device()
    queueNames = []

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    rgbOut = pipeline.create(dai.node.XLinkOut)
    depthOut = pipeline.create(dai.node.XLinkOut)

    rgbOut.setStreamName("rgb")
    queueNames.append("rgb")
    depthOut.setStreamName("depth")
    queueNames.append("depth")

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setFps(fps)
    if downscaleColor:
        camRgb.setIspScale(2, 3)
    # For now, RGB needs fixed focus to properly align with depth.
    # This value was used during calibration
    try:
        calibData = device.readCalibration2()
        lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
        if lensPosition:
            camRgb.initialControl.setManualFocus(lensPosition)
    except:
        raise

    left.setResolution(monoResolution)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    left.setFps(fps)
    right.setResolution(monoResolution)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    right.setFps(fps)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Linking
    camRgb.isp.link(rgbOut.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(depthOut.input)

    # Load model.
    engine = get_engine(args.model)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    input_shape = tuple(map(int, args.input_shape.split(",")))

    elapsed_list = []

    # Connect to device and start pipeline
    with device:
        device.startPipeline(pipeline)

        w, h = camRgb.getIspSize()

        color_frame = None
        depth_data = None

        hostSpatials = HostSpatialsCalc(device)
        delta = 5
        hostSpatials.setDeltaRoi(delta)

        while True:
            latestPacket = {}
            latestPacket["rgb"] = None
            latestPacket["depth"] = None

            queueEvents = device.getQueueEvents(("rgb", "depth"))
            for queueName in queueEvents:
                packets = device.getOutputQueue(queueName).tryGetAll()
                if len(packets) > 0:
                    latestPacket[queueName] = packets[-1]

            if latestPacket["depth"] is not None:
                depth_data = latestPacket["depth"]

            if latestPacket["rgb"] is not None:
                color_frame = latestPacket["rgb"].getCvFrame()

                resized_im = cv2.resize(color_frame, input_shape)
                normalized_im = normalize(resized_im)
                normalized_im = np.expand_dims(normalized_im, axis=0)

                # inference.
                start = time.perf_counter()
                inputs[0].host = normalized_im
                trt_outputs = common.do_inference_v2(
                    context,
                    bindings=bindings,
                    inputs=inputs,
                    outputs=outputs,
                    stream=stream,
                )
                inference_time = (time.perf_counter() - start) * 1000

                num_detections = trt_outputs[0][0]
                boxs = trt_outputs[1].reshape([-1, 4])
                for index in range(num_detections):
                    if trt_outputs[2][index] < args.scoreThreshold:
                        continue

                    # Draw bounding box.
                    class_id = int(trt_outputs[3][index])
                    score = trt_outputs[2][index]
                    box = boxs[index]

                    ymin = int(box[0] * h / input_shape[1])
                    xmin = int(box[1] * w / input_shape[0])
                    ymax = int(box[2] * h / input_shape[1])
                    xmax = int(box[3] * w / input_shape[0])
                    center_x = (xmax - xmin) // 2 + xmin
                    center_y = (ymax - ymin) // 2 + ymin

                    if depth_data is not None:
                        spatials, centroid = hostSpatials.calc_spatials(
                            depth_data, (center_x, center_y)
                        )
                        depth_x = (
                            spatials["x"] / 1000
                            if not math.isnan(spatials["x"])
                            else "--"
                        )
                        depth_y = (
                            spatials["y"] / 1000
                            if not math.isnan(spatials["x"])
                            else "--"
                        )
                        depth_z = (
                            spatials["z"] / 1000
                            if not math.isnan(spatials["y"])
                            else "--"
                        )

                        caption = (
                            "{0}({1:.2f}), x:{2:.2f}, x:{3:.2f}, x:{4:.2f})".format(
                                labels[class_id], score, depth_x, depth_y, depth_z
                            )
                        )
                    else:
                        caption = "{0}({1:.2f})".format(labels[class_id], score)

                    draw_rectangle(
                        color_frame, (xmin, ymin, xmax, ymax), colors[class_id]
                    )
                    draw_caption(color_frame, (xmin, ymin - 10), caption)

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
                draw_caption(color_frame, (10, 30), display_text)

            # Display
            cv2.imshow(WINDOW_NAME, color_frame)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
