# TensorRT PoseNet

## Description
This sample contains code that convert TensorFlow Lite PoseNet model to ONNX model and performs TensorRT inference on Jetson.
1. Download TensorFlow Lite PoseNet Model.
2. Convert to ONNX Model.
3. Convert ONNX Model to Serialize engine and inference on Jetson.

## Environment
- Host PC
  - Linux (Ubuntu 18.04)
- Jetson
  - JetPack 4.5.1

## Convert ONNX Model on your Host PC

### Download TensorFlow Lite model
Download PoseNet's TensorFlow Lite Model from the For TensorFlow Hub.  
`posenet_mobilenet_float_075_1_default_1.tflite`
- https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1


### Convert ONNX Model

Install onnxruntime and tf2onnx.
```
pip3 install onnxruntime tf2onnx
```

Convert TensorFlow Lite Model to ONNX Model.  
```
python3 -m tf2onnx.convert --opset 13 \
    --tflite ./posenet_mobilenet_float_075_1_default_1.tflite \
    --output ./posenet_mobilenet_float_075_1_default_1.onnx
```

## Run Jetson Nano

The following is executed on Jetson (JetPack 4.5.1).

### Install dependency
Install pycuda.  
See details:
- [pycuda installation failure on jetson nano - NVIDIA FORUMS](https://forums.developer.nvidia.com/t/pycuda-installation-failure-on-jetson-nano/77152/22)
```
sudo apt install python3-dev
pip3 install --user cython
pip3 install --global-option=build_ext --global-option="-I/usr/local/cuda/include" --global-option="-L/usr/local/cuda/lib64" pycuda
```

### Clone this repository.
Clone repository.
```
cd ~
git clone https://github.com/NobuoTsukamoto/tensorrt-examples
cd tensorrt-examples
git submodule update --init --recursive
```

### Convert to Serialize engine file.
Copy `posenet_mobilenet_float_075_1_default_1.onnx` to jetson and check model.
```
/usr/src/tensorrt/bin/trtexec --onnx=/home/jetson/tensorrt-examples/models/posenet_mobilenet_float_075_1_default_1.onnx
```

If you want to convert to FP16 model, add --fp16 to the argument of `convert_onnxgs2trt.py`.
```
cd ~/tensorrt-examples/python/detection/
python3 convert_onnxgs2trt.py \
    --model /home/jetson/tensorrt-examples/models/posenet_mobilenet_float_075_1_default_1.onnx \
    --output /home/jetson/tensorrt-examples/models/posenet_mobilenet_float_075_1_default_1_fp16.trt \
    --fp16
```

Finally you can run the demo.
```
python3 trt_simgle_posenet.py \
    --model ../../models/posenet_mobilenet_float_075_1_default_1_fp16.trt
```
