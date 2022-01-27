# TensorRT Examples

## About
TensorRT examples (Jetson, Python/C++)

## List of samples

|Object detection|Pose estimation|
|:--|:--|
|[TensorFlow Lite to TensorRT SSDLite](python/detection/README.md)<br>[EfficientDet-Lite C++ CMake Examples in TensorRT](cpp/efficientdet/README.md)|[TensorFLow Lite to TensorRT PoseNet](python/posenet/README.md)
|![detection](images/detection.gif)|![posenet](images/posenet.gif)|

|MIRNet|ESRGAN|
|:--|:--|
|[TensorFlow Lite to TensorRT MIRNet](python/mirnet/README.md)|[ONNX to TensorRT ESRGAN](python/esrgan/README.md)|[ONNX to TensorRT ESRGAN](python/esrgan/README.md)|
|![mirnet](images/mirnet.gif)|![esrgan](images/esrgan.png)|![esrgan](images/esrgan.png)|

|U^2-Net|
|:--|
|[ONNX to TensorRT U^2-Net](python/u2net/README.md)|
|![u^2-net](images/u2net.gif)|

|Face landmark|Yu-Net|
|:--|:--|
|[ONNX to TensorRT Face landmark](python/face_landmark/README.md)|[ONNX to TensorRT Yu-Net](python/yunet/README.md)|
|![face-landmark](images/keypoint.gif)|![Yu-Net](images/yunet.gif)|

| DeepLab v3+ EdgeTPUV2 and AutoSeg EdgeTPU|
|:--|
|[Convert ONNX Model and otimize the model  using openvino2tensorflow and tflite2tensorflow.](python/deeplabv3_edgetpuv2/README.md)|
|YouTube Video Link<br>[![](https://img.youtube.com/vi/EDffgHSg11A/0.jpg)](https://youtu.be/EDffgHSg11A)|

|Ultra-Fast-Lane-Detection|Fast-SCNN|
|:--|:--|
|[ONNX to TensorRT Ultra-Fast-Lane-Detection](python/ultra_fast_lane_detection/README.md)|[ONNX to TensorRT Fast-SCNN](python/fast_scnn/README.md)|
|YouTube Video Link<br>[![](https://img.youtube.com/vi/gsqi37XZF9M/0.jpg)](https://youtu.be/gsqi37XZF9M)|YouTube Video Link<br>[![](https://img.youtube.com/vi/Lg6BvEgN9AA/0.jpg)](https://youtu.be/Lg6BvEgN9AA)|

## LICENSE
The following files are licensed under [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT).
- [common.py](python/detection/common.py)
  
## Reference
- [TensorRT](https://github.com/NVIDIA/TensorRT)
- [ONNX](https://github.com/onnx/onnx)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [tf2onnx](https://github.com/onnx/tensorflow-onnx)
- [TensorRT Backend For ONNX](https://github.com/onnx/onnx-tensorrt)
- [TensorFlow Model Garden](https://github.com/tensorflow/models)