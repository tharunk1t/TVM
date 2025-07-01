import time
import tracemalloc
import numpy as np
import onnxruntime as ort
from tvm.contrib import graph_executor
import tvm

def benchmark_onnx(model_path="yolov8n.onnx", input_shape=(1, 3, 640, 640)):
    # ONNX inference
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.rand(*input_shape).astype("float32")

    # Warm-up
    session.run(None, {input_name: dummy_input})

    # Benchmark
    tracemalloc.start()
    start = time.time()
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
    end = time.time()
    mem_peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    print(f"ONNXRuntime ⏱️ Avg Time: {(end - start) / 10:.4f}s | 💾 Peak Mem: {mem_peak:.2f} MB")

def benchmark_tvm(lib_path="tvm_lib/compiled_model.so", input_shape=(1, 3, 640, 640)):
    # TVM inference
    lib = tvm.runtime.load_module(lib_path)
    module = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    dummy_input = np.random.rand(*input_shape).astype("float32")
    module.set_input("images", dummy_input)
    module.run()  # Warm-up

    # Benchmark
    tracemalloc.start()
    start = time.time()
    for _ in range(10):
        module.set_input("images", dummy_input)
        module.run()
    end = time.time()
    mem_peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    print(f"TVM ⏱️ Avg Time: {(end - start) / 10:.4f}s | 💾 Peak Mem: {mem_peak:.2f} MB")

if __name__ == "__main__":
    benchmark_onnx()
    benchmark_tvm()
