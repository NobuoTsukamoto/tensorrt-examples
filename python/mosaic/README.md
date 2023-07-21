# TensorRT MOSAIC

## Description
This sample contains code that performs TensorRT inference.
1. Export TF-Lite.
2. Convert TF-Lite Model to ONNX Model and add argmax or fused argmax.
3. Convert ONNX Model to Serialize engine and inference on Jetson.

## Reference
- [TensorFlow Official Models - MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded Context](https://github.com/tensorflow/models/tree/master/official/projects/mosaic)

## Howto

### Export TF-Lite with argmax
Reference
- https://github.com/NobuoTsukamoto/models/tree/master/official/projects/mosaic/serving#export-tf-lite-fp32-with-argmax

Download ckpt (Current best config).
- https://github.com/tensorflow/models/tree/master/official/projects/mosaic#results

Export TF-Lite FP32.
```
python3 serving/export_tflite.py \
    --model_name=mosaic_mnv35_cityscapes \
    --ckpt_path=MobileNetMultiAVGSeg-r1024-ebf64-gp/gcs_ckpt/best_ckpt-857 \
    --output_dir=/tmp \
    --image_height=1024 \
    --image_width=2048 \
    --finalize_method=resize1024_2048
```

### Convert TF-Lite to ONNX Model
```
python3 -m tf2onnx.convert \
    --opset 13 \
    --tflite /tmp/mosaic_mnv35_cityscapes.tflite \
    --output ./mosaic_mnv35_cityscapes.onnx \
    --inputs-as-nchw serving_default_input_2:0 \
    --dequantize
```
### Add argmax or fused argmax.
Add `argmax`.
```
python3 add_mosaic_argmax.py \
    --input ./mosaic_mnv35_cityscapes.onnx \
    --output ./mosaic_mnv35_cityscapes_argmax.onnx \
```

Add `fused argmax`.
```
python3 add_mosaic_argmax.py \
    --input ./mosaic_mnv35_cityscapes.onnx \
    --output ./mosaic_mnv35_cityscapes_fused_argmax.onnx \
    --fused_argmax
```

Check `trtexec`
```
trtexec --onnx=./mosaic_mnv35_cityscapes_fused_argmax.onnx
```

### Convert ONNX Model to TensorRT Serialize engine file.
Convert to Serialize engine file.
If you want to convert to FP16 model, add `--fp16` to the argument of `convert_onnxgs2trt.py`.
```
cd ~/tensorrt-examples/python/utils
python3 convert_onnxgs2trt.py \
    --model  mosaic_mnv35_cityscapes_fused_argmax.onnx \
    --output mosaic_mnv35_cityscapes_fused_argmax_fp16.trt \
    --fp16
```

Finally you can run the demo.
```
python3 trt_mosaic_capture.py \
    --model mosaic_mnv35_cityscapes_fused_argmax_fp16.trt

or 

python3 trt_mosaic_image.py \
    --model mosaic_mnv35_cityscapes_fused_argmax_fp16 \
    --input input_image.png \
    --output output_image.png
```
