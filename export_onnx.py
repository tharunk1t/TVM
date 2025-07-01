from ultralytics import YOLO

# Export YOLOv8n model to ONNX format
def export_to_onnx(model_path="yolov8n.pt", export_path="yolov8n.onnx"):
    model = YOLO(model_path)
    model.export(format="onnx")
    print(f"✅ Model exported to {export_path}")

if __name__ == "__main__":
    export_to_onnx()
