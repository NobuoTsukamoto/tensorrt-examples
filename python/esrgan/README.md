# TensorRT ESRGAN

## Description
This sample contains code that convert TensorFlow Lite ESRGAN model to ONNX model and performs TensorRT inference on TensorRT Container.
1. Pull TensorRT Container and run container.
2. Download ONNX MIRNet Model from PINTO_model_zoo.
3. Convert ONNX Model to Serialize engine and inference.

## Reference
- [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)

## Environment
- PC
  - NVIDIA GPU
  - Linux
  - Docker or Podman
  - [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

## Pull and run NVIDIA TensorRT container
Run the NVIDIA Tensor RT container in Docker or Podman.  
Requires TensorRT 8 or above (8/21 21.07-py3).
```
sudo podman run -it --rm nvcr.io/nvidia/tensorrt:21.07-py3
```

Install OpenCV-Python
```
apt update
apt install python3-opencv
```

## Download ONNC model
Clone PINTO_model_zoo repository and download MIRNet model.
```
git clone https://github.com/PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/077_ESRGAN/
./download.sh
```

Check `trtexec`
```
trtexec --onnx=./saved_model_INPUT_SIZE/model_float32.onnx
```

## Convert to Serialize engine file.
```
cd /workspace/
git clone https://github.com/NobuoTsukamoto/tensorrt-examples
```

If you want to convert to FP16 model, add --fp16 to the argument of `convert_onnxgs2trt.py`.
```
cd ~/tensorrt-examples/python/utils/
python3 convert_onnxgs2trt.py \
    --model /workspace/PINTO_model_zoo/077_ESRGAN/saved_model_INPUT_SIZE/model_float32.onnx \
    --output /workspace/tensorrt-examples/models/esrgan_INPUT_SIZE.trt \
    --fp16
```

## Run demo

### Clone this repository.

Finally you can run the demo.
```
python3 trt_esrgan.py \
    --model /workspace/tensorrt-examples/models/esrgan_INPUT_SIZE.trt \
    --image INPUT_IMAGE_FILE_PATH \
    --output OUTPUT_IMAGE_FILE_PATH
```
