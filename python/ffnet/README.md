# TensorRT FFNet

## Reference
- [FFNet: Simple and Efficient Architectures for Semantic Segmentation](https://github.com/Qualcomm-AI-research/FFNet)
- [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- [onnx2tf](https://github.com/PINTO0309/onnx2tf)

### Convert ONNX Model to TensorRT Serialize engine file.
Convert to Serialize engine file.
If you want to convert to FP16 model, add `--fp16` to the argument of `convert_onnxgs2trt.py`.
```
cd ~/tensorrt-examples/python/utils
python3 convert_onnxgs2trt.py \
    --model segmentation_ffnet46NS_CCC_mobile_pre_down_fused_argmax.onnx \
    --output ./segmentation_ffnet46NS_CCC_mobile_pre_down_fused_argmax.onnx
```

Finally you can run the demo.
```
python trt_ffnet_capture.py \
    --model ./segmentation_ffnet46NS_CCC_mobile_pre_down_fused_argmax.trt \
    --input_shape 512,1024 \
    --output_shape 512,1024

or 

python3 trt_ffnet_image.py \
    --model ./segmentation_ffnet46NS_CCC_mobile_pre_down_fused_argmax.trt \
    --input __PATH_TO_INPUT_IMAGE_FILE__ \
    --input_shape 512,1024 \
    --output_shape 512,1024
```
