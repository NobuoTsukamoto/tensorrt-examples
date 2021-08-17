# TensorRT MIRNet

## Description
This sample contains code that convert TensorFlow Lite MIRNet model to ONNX model and performs TensorRT inference on TensorRT Container.
1. Pull TensorRT Container and run container.
2. Download TensorFlow Lite MIRNet Model from PINTO_model_zoo.
3. Convert to ONNX Model.
4. Convert ONNX Model to Serialize engine and inference.

## Reference
- [soumik12345/MIRNet](https://github.com/soumik12345/MIRNet)
- [sayakpaul/MIRNet-TFLite-TRT](https://github.com/sayakpaul/MIRNet-TFLite-TRT)
- [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)

## Environment
- PC
  - NVIDIA GPU
  - Linux
  - Docker or Podman
  - [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

## Pull and run NVIDIA TensorRT container
Run the NVIDIA Tensor RT container in Docker or Podman.  
Requires TensorRT 8 or above (8/26 21.07-py3).
```
sudo podman run -it --rm nvcr.io/nvidia/tensorrt:21.07-py3
```

Install OpenCV-Python
```
apt update
apt install python3-opencv
```

## Convert ONNX Model

### Download TensorFlow Lite model
Clone PINTO_model_zoo repository and download MIRNet model.
```
git clone https://github.com/PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/079_MIRNet/
./download_INPUT_SIZE.sh
```

### Convert ONNX Model

Install onnxruntime and tf2onnx.
```
pip3 install onnxruntime tf2onnx tensorflow
```

Convert TensorFlow Lite Model to ONNX Model.  
```
python3 -m tf2onnx.convert --opset 13 \
    --tflite ./mirnet_INPUT_SIZE_float32.tflite \
    --output ./mirnet_INPUT_SIZE_float32.onnx
```

Check `trtexec`
```
trtexec --onnx=./mirnet_INPUT_SIZE_float32.onnx
```

## Run demo

### Clone this repository.
```
cd /workspace/
git clone https://github.com/NobuoTsukamoto/tensorrt-examples
```

### Convert to Serialize engine file.
If you want to convert to FP16 model, add --fp16 to the argument of `convert_onnxgs2trt.py`.
```
cd ~/tensorrt-examples/python/utils/
python3 convert_onnxgs2trt.py \
    --model /workspace/PINTO_model_zoo/079_MIRNet/mirnet_INPUT_SIZE_float32.onnx \
    --output /workspace/tensorrt-examples/models/mirnet_INPUT_SIZE.trt \
    --fp16
```

Finally you can run the demo.
```
python3 trt_mirnet_image.py \
    --model /workspace/tensorrt-examples/models/mirnet_INPUT_SIZE.trt
    --image PATH_TO_INPUT_IMAGE_FILE
    --output PATH_TO_OUTPUT_IMAGE_FILE
    --input_shape INPUT_SIZE(ex. 416,416)
```
