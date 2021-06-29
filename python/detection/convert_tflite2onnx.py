#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Convert tensorflow lite object detection model to onnx model.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""


import argparse

import numpy as np

import onnx
import onnx.numpy_helper
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import Constant
from onnx_graphsurgeon.exporters.onnx_exporter import OnnxExporter


def append_nms(
    graph,
    max_classes_per_detection,
    max_detections,
    back_ground_Label_id,
    nms_iou_threshold,
    nms_score_threshold,
    num_classes,
    y_scale,
    x_scale,
    h_scale,
    w_scale,
):
    # https://github.com/NVIDIA/TensorRT/issues/795
    # out_tensors = graph.outputs
    tmap = graph.tensors()
    boxes_input = tmap["concat"]
    scores_input = tmap["convert_scores"]
    anchors = tmap["anchors"]

    onnx_tensor = OnnxExporter.export_tensor_proto(anchors)
    anchors_value = onnx.numpy_helper.to_array(onnx_tensor)
    anchors_value = np.expand_dims(anchors_value, 0)
    anchors_input = Constant(name="anchors_input", values=anchors_value)

    batch_size = 1
    keep_top_k = 10

    nms_attrs = {
        "max_classes_per_detection": max_classes_per_detection,
        "max_detections": max_detections,
        "back_ground_Label_id": back_ground_Label_id,
        "nms_iou_threshold": nms_iou_threshold,
        "nms_score_threshold": nms_score_threshold,
        "num_classes": num_classes,
        "y_scale": y_scale,
        "x_scale": x_scale,
        "h_scale": h_scale,
        "w_scale": w_scale,
        "scoreBits": 16,
    }

    print(nms_attrs)

    num_detections = gs.Variable(
        name="num_detections", dtype=np.int32, shape=(batch_size, 1)
    )
    boxes = gs.Variable(
        name="nmsed_boxes", dtype=np.float32, shape=(batch_size, keep_top_k, 4)
    )
    scores = gs.Variable(
        name="nmsed_scores", dtype=np.float32, shape=(batch_size, keep_top_k)
    )
    classes = gs.Variable(
        name="nmsed_classes", dtype=np.float32, shape=(batch_size, keep_top_k)
    )
    graph.outputs = [num_detections, boxes, scores, classes]

    nms = gs.Node(
        op="TFLiteNMS_TRT",
        attrs=nms_attrs,
        inputs=[boxes_input, scores_input, anchors_input],
        outputs=[num_detections, boxes, scores, classes],
    )
    graph.nodes.append(nms)

    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_classes_per_detection", type=int, default=1)
    parser.add_argument("--max_detections", type=int, default=10)
    parser.add_argument("--back_ground_Label_id", type=int, default=0)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.6)
    parser.add_argument("--nms_score_threshold", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=91)
    parser.add_argument("--y_scale", type=float, default=10.0)
    parser.add_argument("--x_scale", type=float, default=10.0)
    parser.add_argument("--h_scale", type=float, default=5.0)
    parser.add_argument("--w_scale", type=float, default=5.0)

    args = parser.parse_args()

    input_model_path = args.input
    output_model_path = args.output

    graph = gs.import_onnx(onnx.load(input_model_path))
    graph = append_nms(
        graph,
        args.max_classes_per_detection,
        args.max_detections,
        args.back_ground_Label_id,
        args.nms_iou_threshold,
        args.nms_score_threshold,
        args.num_classes,
        args.y_scale,
        args.x_scale,
        args.h_scale,
        args.w_scale,
    )

    # Remove unused nodes, and topologically sort the graph.
    graph.cleanup().toposort().fold_constants().cleanup()

    # Export the onnx graph from graphsurgeon
    onnx.save_model(gs.export_onnx(graph), output_model_path)

    print("Saving the ONNX model to {}".format(output_model_path))


if __name__ == "__main__":
    main()
