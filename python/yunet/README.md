# TensorRT Yu-Net

## Description
This sample contains code that performs TensorRT inference on Jetson.
1. Download ONNX U^2-Net Model from PINTO_model_zoo.
2. Convert ONNX Model to Serialize engine and inference on Jetson.

## Reference
- [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/dev)

## Environment
- Jetson
  - JetPack 4.6

## Download ONNX Model on your Jetson

Clone PINTO_model_zoo repository and download MIRNet model.
```
git clone https://github.com/PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/144_YuNet
./download.sh
```

Check `trtexec`
```
/usr/src/tensorrt/bin/trtexec --onnx=./saved_model/face_detection_yunet_120x160.onnx
```

## Run Jetson Nano

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
Copy `face_detection_yunet_120x160.onnx` to `tensorrt-examples/models`.
```
cp ~/PINTO_model_zoo/saved_model/face_detection_yunet_120x160.onnx ~/tensorrt-examples/models/
```

```
cd ~/tensorrt-examples/python/utils
python3 convert_onnxgs2trt.py \
    --model /home/jetson/tensorrt-examples/models/face_detection_yunet_120x160.onnx \
    --output /home/jetson/tensorrt-examples/models/face_detection_yunet_120x160.trt \
```

Finally you can run the demo.
```
python3 trt_yunet_capture.py \
    --model ../../models/face_detection_yunet_120x160.trt
```
