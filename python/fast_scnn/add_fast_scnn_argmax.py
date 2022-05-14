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


def append_argmax(graph):
    print("Add ArgMax.")

    batch, _, h, w = graph.outputs[0].shape
    argmax_inputs = [node for node in graph.nodes if node.op == "Resize"][-1].outputs

    argmax_attrs = {
        "axis": 1,
        "keepdims": 1,
        "select_last_index": False,
    }

    output = gs.Variable(name="output", dtype=np.int64, shape=(batch, h, w))
    graph.outputs = [output]

    argmax_node = gs.Node(
        op="ArgMax",
        attrs=argmax_attrs,
        inputs=argmax_inputs,
        outputs=[output],
    )
    graph.nodes.append(argmax_node)

    return graph


def append_fused_argmax(graph):
    print("Add Fused ArgMax.")

    batch, num_class, h, w = graph.outputs[0].shape

    # Add resize ope
    conv_outputs = [node for node in graph.nodes if node.op == "Conv"][-1].outputs[0]
    resize_node = [node for node in graph.nodes if node.op == "Resize"][-1]
    resize_roi_input = resize_node.inputs[1]
    resize_scales_input = resize_node.inputs[2]
    resize_size_input = gs.Constant(
        "resize_size",
        values=np.array([batch, num_class, h // 2, w // 2], dtype=np.int64),
    )
    resize_attrs = {
        "coordinate_transformation_mode": resize_node.attrs[
            "coordinate_transformation_mode"
        ],
        "cubic_coeff_a": resize_node.attrs["cubic_coeff_a"],
        # "exclude_outside": resize_node.attrs["exclude_outside"],
        # "extrapolation_value": resize_node.attrs["extrapolation_value"],
        "mode": resize_node.attrs["mode"],
        "nearest_mode": resize_node.attrs["nearest_mode"],
    }
    resize_output = gs.Variable(
        name="resize_output",
        dtype=np.float32,
        shape=(batch, num_class, h // 2, w // 2),
    )
    resize_node = gs.Node(
        op="Resize",
        attrs=resize_attrs,
        inputs=[conv_outputs, resize_roi_input, resize_scales_input, resize_size_input],
        outputs=[resize_output],
    )
    graph.nodes.append(resize_node)

    # Add Argmax
    argmax_attrs = {
        "axis": 1,
        "keepdims": 1,
        "select_last_index": False,
    }
    argmax_output = gs.Variable(
        name="argmax_output", dtype=np.int64, shape=(batch, 1, h // 2, w // 2)
    )
    argmax_node = gs.Node(
        op="ArgMax",
        attrs=argmax_attrs,
        inputs=[resize_output],
        outputs=[argmax_output],
    )
    graph.nodes.append(argmax_node)

    # Cast INT64 to Float32
    cast_attrs = {
        "to": int(onnx.TensorProto.FLOAT),
    }
    cast_output = gs.Variable(
        name="cast_output", dtype=np.float32, shape=(batch, 1, h // 2, w // 2)
    )
    cast_node = gs.Node(
        op="Cast",
        attrs=cast_attrs,
        inputs=[argmax_output],
        outputs=[cast_output],
    )
    graph.nodes.append(cast_node)

    # Add Resize Nearest
    nearest_roi_input = gs.Constant(
        "resize_nearest_roi", values=np.array([], dtype=np.float32)
    )
    nearest_scale_input = gs.Constant(
        "resize_nearest_scale", values=np.array([], dtype=np.float32)
    )
    nearest_size_input = gs.Constant(
        "resize_nearest_size", values=np.array([batch, 1, h, w], dtype=np.int64)
    )
    resize_nearest_attrs = {
        "coordinate_transformation_mode": resize_node.attrs[
            "coordinate_transformation_mode"
        ],
        "cubic_coeff_a": resize_node.attrs["cubic_coeff_a"],
        "exclude_outside": 0,
        "mode": "nearest",
        "nearest_mode": "round_prefer_ceil",
    }
    neaest_output = gs.Variable(
        name="fused_argmax_output",
        dtype=np.float32,
        shape=(batch, h, w),
    )
    resize_nearestniare_node = gs.Node(
        op="Resize",
        attrs=resize_nearest_attrs,
        inputs=[
            cast_output,
            nearest_roi_input,
            nearest_scale_input,
            nearest_size_input,
        ],
        outputs=[neaest_output],
    )
    graph.nodes.append(resize_nearestniare_node)

    # Cast Float32 to INT32
    output_cast_attrs = {
        "to": int(onnx.TensorProto.INT64),
    }
    output = gs.Variable(
        name="output",
        dtype=np.int64,
        shape=(batch, 1, h, w),
    )
    graph.outputs = [output]
    output_cast_node = gs.Node(
        op="Cast",
        attrs=output_cast_attrs,
        inputs=[neaest_output],
        outputs=[output],
    )
    graph.nodes.append(output_cast_node)

    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Input ONNX model path."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output ONXX (Add TFLIteNMS_TRT) model path.",
    )
    parser.add_argument("--fused_argmax", action="store_true")
    args = parser.parse_args()

    input_model_path = args.input
    output_model_path = args.output

    graph = gs.import_onnx(onnx.load(input_model_path))
    if args.fused_argmax:
        graph = append_fused_argmax(graph)
    else:
        graph = append_argmax(graph)

    # Remove unused nodes, and topologically sort the graph.
    graph.cleanup().toposort().fold_constants().cleanup()

    # Export the onnx graph from graphsurgeon
    onnx.save_model(gs.export_onnx(graph), output_model_path)

    print("Saving the ONNX model to {}".format(output_model_path))


if __name__ == "__main__":
    main()
