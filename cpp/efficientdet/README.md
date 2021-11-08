# EfficientDet-Lite C++ CMake Examples in TensorRT.

This sample contains a sample to run EfficientDet-Lite on Jetson Nano using the EfficientNMS plugin.

## Reference
- [EfficientDet Object Detection in TensorRT](https://github.com/NVIDIA/TensorRT/tree/main/samples/python/efficientdet)
- [Efficient NMS Plugin TensorRT](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin)
- [EfficientDet Google AutoML](https://github.com/google/automl/tree/master/efficientdet)

## How to

### Convert AutoML Model to ONNX Model
This notebook contains a sample that converts EfficientDet-Lite's AutoML Model into an ONNX model for running on TensorRT.  
Convert to ONNX model with Google Colab or Host PC and copy ONNX model to Jetson Nano.
- [TensorRT EfficientDet-Lite Model Conversion AutoML Models to ONNX Model](Export_EfficientDetLite_TensorRT.ipynb)

### Build and run Jetson Nano
#### Install dependency packages.
```
$ sudo apt install libopencv-dev cmake libboost-dev
```

#### Clone repository and init submodule.
```
$ cd ~
$ git clone https://github.com/NobuoTsukamoto/edge_tpu.git
$ cd edge_tpu
$ git submodule init && git submodule update
```

### Convert ONNX model to TensorRT engine.
When converting EfficientDet-Lite0.  
Model path:  `~/tensorrt-examples/cpp/efficientdet/efficientdet-lite4.onnx`
```
$ cd ~/tensorrt-examples/TensorRT/samples/python/efficientdet/
$ python3 build_engine.py \
    --onnx ~/tensorrt-examples/cpp/efficientdet/efficientdet-lite0.onnx \
    --engine ~/tensorrt-examples/cpp/efficientdet/efficientdet-lite0.trt
```

### Build sample.
```
$ cd ~/tensorrt-examples/cpp/efficientdet/
$ mkdir build && cd build
$ cmake ..
$ make
```
## Run sample on Jetson Nano
```
$ ./trt_efficientdet \
    ~/tensorrt-examples/cpp/efficientdet/efficientdet-lite0.trt \
    --width=320 \
    --height=320 \
    --label=~/tensorrt-examples/models/coco_labels.txt
```

## Usage
```
$ ./trt_efficientdet --help
Usage: trt_efficientdet [params] input 

	-?, -h, --help, --usage (value:true)
		show help command.
	-H, --height (value:512)
		input model height.
	-f, --file
		path to video file.
	-l, --label (value:.)
		path to label file.
	-o, --output
		output video file path.
	-s, --score (value:0.5)
		score threshold.
	-w, --width (value:512)
		input model width.

	input
		path to trt engine file.
```
