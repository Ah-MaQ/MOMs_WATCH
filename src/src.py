import onnx

# Load the ONNX model
model_path = "model.onnx"  # 로컬 환경에 맞게 경로를 설정하세요.
onnx_model = onnx.load(model_path)

# Check the model for validity
onnx.checker.check_model(onnx_model)

# Extract and print the model's input and output details
input_details = []
output_details = []

for input_tensor in onnx_model.graph.input:
    input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    input_details.append((input_tensor.name, input_shape))

for output_tensor in onnx_model.graph.output:
    output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
    output_details.append((output_tensor.name, output_shape))

print("Model Inputs:", input_details)
print("Model Outputs:", output_details)