#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Convert TensorFlow Lite Object detection ONNX Model to TRT engine.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import os

import tensorrt as trt

import common

TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def build_engine(onnx_file_path, output_engine_file_path, fp16):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        common.EXPLICIT_BATCH
    ) as network, builder.create_builder_config() as config, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser:
        config.max_workspace_size = 1 << 28  # 256MiB
        builder.max_batch_size = 1
        if fp16:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            config.set_flag(trt.BuilderFlag.FP16)

        # Parse model file
        print("Loading ONNX file from path {}...".format(onnx_file_path))
        if not os.path.exists(onnx_file_path):
            print("ONNX file {} not found.".format(onnx_file_path))
            return

        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return

        print("Completed parsing of ONNX file")
        print(
            "Building an engine from file {}; this may take a while...".format(
                onnx_file_path
            )
        )
        engine = builder.build_engine(network, config)
        print("Completed creating Engine")
        with open(output_engine_file_path, "wb") as f:
            f.write(engine.serialize())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of onnx model.", required=True)
    parser.add_argument(
        "--output", help="File path of output trt model.", required=True
    )
    parser.add_argument("--fp16", help="Enable FP16.", action="store_true")
    args = parser.parse_args()

    build_engine(args.model, args.output, args.fp16)


if __name__ == "__main__":
    main()
