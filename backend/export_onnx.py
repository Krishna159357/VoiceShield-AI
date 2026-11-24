import inspect
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
FP32_PATH = OUT_DIR / "wav2vec2_xlsr_fp32.onnx"
INT8_PATH = OUT_DIR / "wav2vec2_xlsr_int8.onnx"


class Wav2Vec2ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        out = self.model(input_values=input_values, output_hidden_states=True)
        emb = out.hidden_states[-1].mean(dim=1)
        return emb, out.logits


def main():
    print(f"[1/4] Loading {MODEL_ID} on CPU...", flush=True)
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    pt_model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).eval()
    wrapper = Wav2Vec2ExportWrapper(pt_model).eval()

    dummy_audio = np.random.randn(32000).astype(np.float32) * 0.1  # 2.0s @ 16kHz
    inputs = extractor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    dummy_input = inputs["input_values"]

    print(f"[2/4] Exporting FP32 ONNX to {FP32_PATH}...", flush=True)
    export_kwargs = dict(
        input_names=["input_values"],
        output_names=["embedding", "logits"],
        dynamic_axes={"input_values": {0: "batch", 1: "sequence"}},
        opset_version=17,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(wrapper, (dummy_input,), str(FP32_PATH), **export_kwargs)

    print(f"[3/4] Applying Dynamic INT8 Quantization -> {INT8_PATH}...", flush=True)
    quantize_dynamic(
        model_input=str(FP32_PATH),
        model_output=str(INT8_PATH),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
    )
    FP32_PATH.unlink(missing_ok=True)  # Remove bulky 1.2GB FP32 temp file

    print("[4/4] Benchmarking PyTorch FP32 vs. ONNX INT8 (5 runs each)...", flush=True)
    with torch.inference_mode():
        t0 = time.perf_counter()
        for _ in range(5):
            pt_emb, _ = wrapper(dummy_input)
        pt_ms = (time.perf_counter() - t0) / 5 * 1000

    sess = ort.InferenceSession(str(INT8_PATH), providers=["CPUExecutionProvider"])
    ort_in = {"input_values": dummy_input.numpy()}
    t0 = time.perf_counter()
    for _ in range(5):
        ort_emb, _ = sess.run(["embedding", "logits"], ort_in)
    ort_ms = (time.perf_counter() - t0) / 5 * 1000

    cos_sim = float(
        np.dot(pt_emb.numpy().flatten(), ort_emb.flatten())
        / (np.linalg.norm(pt_emb.numpy()) * np.linalg.norm(ort_emb) + 1e-8)
    )
    print(f"\n=== Phase 1 Benchmark Results ===")
    print(f"PyTorch FP32 : {pt_ms:.1f} ms / 2s window")
    print(f"ONNX INT8    : {ort_ms:.1f} ms / 2s window ({pt_ms / ort_ms:.2f}x faster!)")
    print(f"Embedding Fidelity (Cosine Sim): {cos_sim:.4f}")
    print(f"Saved quantized model: {INT8_PATH} ({INT8_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()