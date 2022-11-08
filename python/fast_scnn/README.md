# TensorRT Fast-SCNN

## Description
This sample contains code that performs TensorRT inference on Jetson.
1. Download ONNX Fast-SCNN Model from PINTO_model_zoo.
2. Convert ONNX Model to Serialize engine and inference on Jetson.

## Reference
- [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
    - [228_Fast-SCNN](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/228_Fast-SCNN)
    - [Demo projects](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/228_Fast-SCNN/demo)

## Howto

### Download ONNX Model

Clone PINTO_model_zoo repository and download Fast-SCNN
 model.
```
git clone https://github.com/PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/228_Fast-SCNN/
./download.sh
```

Check `trtexec`
```
/usr/src/tensorrt/bin/trtexec --onnx=./fast_scnn_NNNxNNN/fast_scnn_NNNxNNN.onnx
```

### Convert ONNX Model to TensorRT Serialize engine file.
Copy `fast_scnn_NNNxNNN.onnx` to `tensorrt-examples/models`.  
In the following, `fast_scnn_576x768.onnx` is taken as an example.
```
cp ~/home/nobuo/Data/models/PINTO_model_zoo/228_Fast-SCNN/fast_scnn_576x768/fast_scnn_576x768.onnx ~/tensorrt-examples/models/
```

Convert to Serialize engine file.
If you want to convert to FP16 model, add `--fp16` to the argument of `convert_onnxgs2trt.py`.
```
cd ~/tensorrt-examples/python/utils
python3 convert_onnxgs2trt.py \
    --model /home/jetson/tensorrt-examples/models/fast_scnn_576x768.onnx \
    --output /home/jetson/tensorrt-examples/models/fast_scnn_576x768.trt \
```

Finally you can run the demo.
```
python3 trt_fast_scnn_capture.py \
    --model ../../models/fast_scnn_576x768.trt
    --input_shape 576,768

or 

python3 trt_fast_scnn_image.py \
    --model ../../models/fast_scnn_576x768.trt
    --input_shape 576,768
    --input input_image.png
    --output output_image.png
```
