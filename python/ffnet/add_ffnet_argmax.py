#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Script to add ArgMax to FFNet.

    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""


import argparse

import numpy as np
import onnx
import onnx.numpy_helper
import onnx_graphsurgeon as gs


def append_resize_argmax(graph):
    print("Add Resize and ArgMax.")

    _, _, height, width = graph.inputs[0].shape
    batch, num_class, _, _ = graph.outputs[0].shape
    print(batch, num_class, height, width)

    # Add resize ope
    conv_outputs = [node for node in graph.nodes if node.op == "Conv"][-1].outputs[0]
    resize_roi_input = gs.Constant("resize_roi", values=np.array([], dtype=np.float32))
    resize_scales_input = gs.Constant("resize_scales", values=np.array([], dtype=np.float32))
    resize_size_input = gs.Constant(
        "resize_size",
        values=np.array([batch, num_class, height, width], dtype=np.int64),
    )
    resize_attrs = {
        "coordinate_transformation_mode": "align_corners",
        "cubic_coeff_a": -0.75,
        "mode": "linear",
        "nearest_mode": "floor",
    }
    resize_output = gs.Variable(
        name="resize_output",
        dtype=np.float32,
        shape=(batch, num_class, height, width),
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
    output = gs.Variable(name="output", dtype=np.int64, shape=(batch, height, width))
    graph.outputs = [output]
    argmax_node = gs.Node(
        op="ArgMax",
        attrs=argmax_attrs,
        inputs=[resize_output],
        outputs=[output],
    )
    graph.nodes.append(argmax_node)

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
        help="Output ONXX (Add ArgMax) model path.",
    )
    args = parser.parse_args()

    input_model_path = args.input
    output_model_path = args.output

    graph = gs.import_onnx(onnx.load(input_model_path))
    graph = append_resize_argmax(graph)

    # Remove unused nodes, and topologically sort the graph.
    graph.cleanup().toposort().fold_constants().cleanup()

    # Export the onnx graph from graphsurgeon
    onnx.save_model(gs.export_onnx(graph), output_model_path)

    print("Saving the ONNX model to {}".format(output_model_path))


if __name__ == "__main__":
    main()
