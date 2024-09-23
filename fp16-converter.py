import onnx
from onnxconverter_common import float16

model = onnx.load("e5_large_fp_32/model.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "model_fp16.onnx")
