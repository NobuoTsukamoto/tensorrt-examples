# TensorRT DeepLab v3+ EdgeTPUV2 and AutoSeg EdgeTPU

## Description
This sample contains code that convert TensorFlow Hub DeepLab v3+ EdgeTPUV2 and AutoSeg EdgeTPU model to ONNX model and performs TensorRT inference on Jetson.
Optimize the DeepLab v3+ EdgeTPUV2 model using [openvino2tensorflow](https://github.com/PINTO0309/openvino2tensorflow) and [tflite2tensorflow](https://github.com/PINTO0309/tflite2tensorflow).
1. Convert TensorFlow Lite model from TensorFlow Hub model.
2. Convert TensorFlow Lite model to ONNX Model.
3. (DeepLab v3+ EdgeTPUV2) Optimize the model using [openvino2tensorflow](https://github.com/PINTO0309/openvino2tensorflow) and [tflite2tensorflow](https://github.com/PINTO0309/tflite2tensorflow).
4. Convert ONNX Model to Serialize engine and inference on Jetson.

## Reference
- [autoseg-edgetpu](https://tfhub.dev/google/collections/autoseg-edgetpu/1)
- [deeplab-edgetpu](https://tfhub.dev/google/collections/deeplab-edgetpu/1)
- [openvino2tensorflow](https://github.com/PINTO0309/openvino2tensorflow)
- [tflite2tensorflow](https://github.com/PINTO0309/tflite2tensorflow)


## Environment
- Host PC
  - Linux or Google Colab
- Jetson
  - JetPack 4.6

## Convert ONNX Model on your Google Colab
[Convert DeepLab v3+ EdgeTPUv2 TF-Hub model to ONNX model Notebook](convert_deeplabv3_edgetpuv2_tfhub2onnx.ipynb) contains all the steps to convert from TensorFlow Hub model to the ONNX model.  
Run it on Google Colab and download the converted ONNX model. Of course, you can also run it on your own host PC.

### Optimize Model
Optimize your model by using [openvino2tensorflow](https://github.com/PINTO0309/openvino2tensorflow) and [tflite2tensorflow](https://github.com/PINTO0309/tflite2tensorflow). Jetson Nano can improve 5ms with the FP16 model of `deeplab-edgetpu_default_argmax_xs`. In the result of trtexec, the latency after optimization can be less than 100ms.


Start the openvino2tensorflow container.  
(I use podman for Fedora 35.)
```
sudo podman run -it --rm -v DOWNLOAD_ONNX_MODEL_DIR:/work ghcr.io/pinto0309/openvino2tensorflow:latest
```

When optimizing `deeplab-edgetpu_default_argmax_xs`:
```
cd /work
MODEL=deeplab-edgetpu_default_argmax_xs
```

Convert TensorFlow Lite Model to OpenVINO Model.
```
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py --input_model ${MODEL}.onnx --data_type FP32 --output_dir saved_model_${MODEL}/openvino/FP32
```

Convert OpenVINO model to Saved Model.
```
openvino2tensorflow \
    --model_path saved_model_${MODEL}/openvino/FP32/${MODEL}.xml \
    --output_pb
pb_to_saved_model \
    --pb_file_path saved_model/model_float32.pb \
    --inputs inputs:0 \
    --outputs model/tf.reshape/Reshape:0
mv saved_model_from_pb/* saved_model
rm -rf saved_model_from_pb
```

Convert Saved Model to TensorFlow Lite Model.
```
saved_model_to_tflite \
    --saved_model_dir_path saved_model \
    --output_no_quant_float32_tflite
mv tflite_from_saved_model/model_float32.tflite saved_model
```

Finaly, Convert TensorFLow Lite Model to ONNX Model.
```
tflite2tensorflow \
    --model_path ./saved_model/model_float32.tflite \
    --schema_path ~/schema.fbs \
    --flatc_path ~/flatc \
    --output_onnx
mv saved_model/model_float32.onnx ${MODEL}_opt.onnx
```

#### Latency before and after optimization with Jetson Nano
Before optimization
```
 /usr/src/tensorrt/bin/trtexec --onnx=/home/jetson/deeplab-edgetpu_default_argmax_xs.onnx --fp16

 ...

[12/02/2021-19:58:58] [I] === Performance summary ===
[12/02/2021-19:58:58] [I] Throughput: 9.85818 qps
[12/02/2021-19:58:58] [I] Latency: min = 100.036 ms, max = 104.225 ms, mean = 101.425 ms, median = 101.092 ms, percentile(99%) = 104.225 ms
[12/02/2021-19:58:58] [I] End-to-End Host Latency: min = 100.049 ms, max = 104.233 ms, mean = 101.438 ms, median = 101.105 ms, percentile(99%) = 104.233 ms
[12/02/2021-19:58:58] [I] Enqueue Time: min = 57.1197 ms, max = 61.1654 ms, mean = 58.9862 ms, median = 58.2306 ms, percentile(99%) = 61.1654 ms
[12/02/2021-19:58:58] [I] H2D Latency: min = 0.307251 ms, max = 0.312866 ms, mean = 0.310417 ms, median = 0.310425 ms, percentile(99%) = 0.312866 ms
[12/02/2021-19:58:58] [I] GPU Compute Time: min = 99.6099 ms, max = 103.805 ms, mean = 101.003 ms, median = 100.67 ms, percentile(99%) = 103.805 ms
[12/02/2021-19:58:58] [I] D2H Latency: min = 0.110596 ms, max = 0.114624 ms, mean = 0.112055 ms, median = 0.111938 ms, percentile(99%) = 0.114624 ms
[12/02/2021-19:58:58] [I] Total Host Walltime: 3.24604 s
[12/02/2021-19:58:58] [I] Total GPU Compute Time: 3.23208 s
[12/02/2021-19:58:58] [I] Explanations of the performance metrics are printed in the verbose logs.
```
After optimization
```
 /usr/src/tensorrt/bin/trtexec --onnx=/home/jetson/deeplab-edgetpu_default_argmax_xs_opt.onnx --fp16

 ...

[12/02/2021-20:16:53] [I] === Performance summary ===
[12/02/2021-20:16:53] [I] Throughput: 10.5582 qps
[12/02/2021-20:16:53] [I] Latency: min = 93.6714 ms, max = 97.5481 ms, mean = 94.7022 ms, median = 94.5459 ms, percentile(99%) = 97.5481 ms
[12/02/2021-20:16:53] [I] End-to-End Host Latency: min = 93.678 ms, max = 97.5593 ms, mean = 94.7129 ms, median = 94.5574 ms, percentile(99%) = 97.5593 ms
[12/02/2021-20:16:53] [I] Enqueue Time: min = 6.27478 ms, max = 7.67041 ms, mean = 6.55616 ms, median = 6.46667 ms, percentile(99%) = 7.67041 ms
[12/02/2021-20:16:53] [I] H2D Latency: min = 0.300903 ms, max = 0.331909 ms, mean = 0.30406 ms, median = 0.303345 ms, percentile(99%) = 0.331909 ms
[12/02/2021-20:16:53] [I] GPU Compute Time: min = 93.26 ms, max = 97.1343 ms, mean = 94.2885 ms, median = 94.1343 ms, percentile(99%) = 97.1343 ms
[12/02/2021-20:16:53] [I] D2H Latency: min = 0.10791 ms, max = 0.116455 ms, mean = 0.109651 ms, median = 0.109375 ms, percentile(99%) = 0.116455 ms
[12/02/2021-20:16:53] [I] Total Host Walltime: 3.12555 s
[12/02/2021-20:16:53] [I] Total GPU Compute Time: 3.11152 s
[12/02/2021-20:16:53] [I] Explanations of the performance metrics are printed in the verbose logs.

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
Copy `deeplab-edgetpu_XXX or autoseg-edgetpu_XXX.onnx` to `tensorrt-examples/models`.

```
cd ~/tensorrt-examples/python/utils
python3 convert_onnxgs2trt.py \
    --model /home/jetson/tensorrt-examples/models/XXX.onnx \
    --output /home/jetson/tensorrt-examples/models/XXX.trt \
    --fp16
```

Finally you can run the demo.
```
python3 trt_deeplabv3_edgetpuv2.py \
    --model ../../models/XXX.trt 
```