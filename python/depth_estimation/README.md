# TensorRT Depth estimation

## Description
This sample contains code that performs TensorRT inference on Jetson.
1. Download ONNX Depth estimation Model from PINTO_model_zoo.
2. Convert ONNX Model to Serialize engine and inference on Jetson.

## Reference
- [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- [alinstein/Depth_estimation](https://github.com/alinstein/Depth_estimation)

## Howto
### Download ONNX Model

Clone PINTO_model_zoo repository and download MIRNet model.
```
git clone https://github.com/PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/149_depth_estimation/
./download.sh
```

Check `trtexec`
```
/usr/src/tensorrt/bin/trtexec --onnx=./saved_model_INPUT_SIZE/depth_estimation_mbnv2_INPUT_SIZE.onnx
```

### Run
Copy `depth_estimation_mbnv2_INPUT_SIZE.onnx` to `tensorrt-examples/models`.
```
cp ~/PINTO_model_zoo/149_depth_estimation/saved_model_INPUT_SIZE/depth_estimation_mbnv2_INPUT_SIZE.onnx ~/tensorrt-examples/models/
```

Convert to Serialize engine file.
If you want to convert to FP16 model, add --fp16 to the argument of `convert_onnxgs2trt.py`.
```
cd ~/tensorrt-examples/python/utils
python3 convert_onnxgs2trt.py \
    --model /home/jetson/tensorrt-examples/models/depth_estimation_mbnv2_INPUT_SIZE.onnx \
    --output /home/jetson/tensorrt-examples/models/depth_estimation_mbnv2_INPUT_SIZE.trt \
```

Finally you can run the demo.
```
python3 trt_depth_estimation_capture.py \
    --model ../../models/depth_estimation_mbnv2_INPUT_SIZE.trt
```
