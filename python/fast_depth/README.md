# TensorRT FastDepth

## Description
This sample contains code that performs TensorRT inference on Jetson.
1. Download ONNX FastDepth Model from PINTO_model_zoo.
2. Convert ONNX Model to Serialize engine and inference on Jetson.

## Reference
- [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- [dwofk/fast-depth](https://github.com/dwofk/fast-depth)

## Howto
### Download ONNX Model

Clone PINTO_model_zoo repository and download MIRNet model.
```
git clone https://github.com/PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/146_FastDepth/
./download.sh
```

Check `trtexec`
```
/usr/src/tensorrt/bin/trtexec --onnx=./saved_model_INPUT_SIZE/fast_depth_INPUT_SIZE.onnx
```

### Run
Copy `fast_depth_INPUT_SIZE.onnx` to `tensorrt-examples/models`.
```
cp ~/PINTO_model_zoo/146_FastDepth/saved_model_INPUT_SIZE/fast_depth_INPUT_SIZE.onnx ~/tensorrt-examples/models/
```

Convert to Serialize engine file.
If you want to convert to FP16 model, add --fp16 to the argument of `convert_onnxgs2trt.py`.
```
cd ~/tensorrt-examples/python/utils
python3 convert_onnxgs2trt.py \
    --model /home/jetson/tensorrt-examples/models/fast_depth_INPUT_SIZE.onnx \
    --output /home/jetson/tensorrt-examples/models/fast_depth_INPUT_SIZE.trt \
```

Finally you can run the demo.
```
python3 trt_fast_depth_capture.py \
    --model ../../models/fast_depth_INPUT_SIZE.trt
```
