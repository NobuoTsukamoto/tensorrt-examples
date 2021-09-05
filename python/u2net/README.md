# TensorRT U^2-Net

## Description
This sample contains code that performs TensorRT inference on Jetson.
1. Download ONNX U^2-Net Model from PINTO_model_zoo.
2. Convert ONNX Model to Serialize engine and inference on Jetson.

## Reference
- [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)

## Environment
- Jetson
  - JetPack 4.6

## Download ONNX Model on your Jetson

Clone PINTO_model_zoo repository and download MIRNet model.
```
git clone https://github.com/PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/061_U-2-Net/30_human_segmentation/
./download_320x320.sh
```

Check `trtexec`
```
/usr/src/tensorrt/bin/trtexec --onnx=./saved_model_320x320/u2net_human_seg_320x320.onnx
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
Copy `u2net_human_seg_320x320.onnx` to `tensorrt-examples/models`.
```
cp ~/PINTO_model_zoo/061_U-2-Net/30_human_segmentation/saved_model_320x320/u2net_human_seg_320x320.onnx ~/tensorrt-examples/models/
```

If you want to convert to FP16 model, add --fp16 to the argument of `convert_onnxgs2trt.py`.
```
cd ~/tensorrt-examples/python/utils
python3 convert_onnxgs2trt.py \
    --model /home/jetson/tensorrt-examples/models/u2net_human_seg_320x320.onnx \
    --output /home/jetson/tensorrt-examples/models/u2net_human_seg_320x320.trt \
    --fp16
```

Finally you can run the demo.
```
python3 trt_u2net.py \
    --model ../../models/u2net_human_seg_320x320.trt
```
