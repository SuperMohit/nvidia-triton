from onnxconverter_common import auto_mixed_precision
import onnx

model = onnx.load("path/to/model.onnx")
model_fp16 = auto_convert_mixed_precision(model, test_data, rtol=0.01, atol=0.001, keep_io_types=True)
onnx.save(model_fp16, "path/to/model_fp16.onnx")
