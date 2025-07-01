from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(input_path="yolov8n.onnx", output_path="yolov8n-int8.onnx"):
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )
    print(f"✅ Quantized model saved as {output_path}")

if __name__ == "__main__":
    quantize_model()
