import onnxruntime as ort

# Load the ONNX model
session = ort.InferenceSession("/root/model_fp16.onnx")

for input_meta in session.get_inputs():
    print(f"Name: {input_meta.name}, Shape: {input_meta.shape}, Type: {input_meta.type}")


# Print output names
for output_meta in session.get_outputs():
    print(f"Output name: {output_meta.name}, Shape: {output_meta.shape}, Type: {output_meta.type}")
