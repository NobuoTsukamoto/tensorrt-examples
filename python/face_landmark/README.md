# TensorRT Face landmark

## Description
This sample contains code that performs TensorRT inference on Jetson.
1. Download ONNX U^2-Net Model from PINTO_model_zoo.
2. Convert ONNX Model to Serialize engine and inference on Jetson.

## Reference
- [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- [610265158/face_landmark](https://github.com/610265158/face_landmark)

## Environment
- Jetson
  - JetPack 4.6

## Download ONNX Model on your Jetson

Clone PINTO_model_zoo repository and download MIRNet model.
```
git clone https://github.com/PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/043_face_landmark/01_float32
./download.sh
```
## Convert ONNX Model

Install onnxruntime and tf2onnx.
```
pip3 install onnxruntime tf2onnx
```

Convert TensorFlow Lite Model to ONNX Model.  
```
python3 -m tf2onnx.convert --opset 13 --tflite ./keypoints.tflite --output ./keypoints.onnx
```

Check `trtexec`
```
/usr/src/tensorrt/bin/trtexec --onnx=./keypoints.onnx
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
Copy `keypoints.onnx` to `tensorrt-examples/models`.
```
cp ~/PINTO_model_zoo/043_face_landmark/01_float32/keypoints.onnx ~/tensorrt-examples/models/
```

```
cd ~/tensorrt-examples/python/utils
python3 convert_onnxgs2trt.py \
    --model /home/jetson/tensorrt-examples/models/keypoints.onnx \
    --output /home/jetson/tensorrt-examples/models/keypoints.trt \
```

Finally you can run the demo.
```
python3 trt_face_landmark_capture.py \
    --model ../../models/keypoints.trt
```
