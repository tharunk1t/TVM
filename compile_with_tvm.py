import onnx
import tvm
from tvm import relay

def compile_model(onnx_path="yolov8n.onnx", target="llvm", output_dir="tvm_lib"):
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    input_shape = (1, 3, 640, 640)
    shape_dict = {"images": input_shape}

    # Convert ONNX to Relay IR
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    # Compile with TVM
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    # Save compiled library
    lib.export_library(f"{output_dir}/compiled_model.so")
    print(f"✅ Model compiled and saved to {output_dir}")

if __name__ == "__main__":
    compile_model()
