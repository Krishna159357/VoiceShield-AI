from pathlib import Path
import joblib
import librosa
import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import AutoFeatureExtractor

MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
CUSTOM_CLF_PATH = Path("models/custom_deepfake_clf.joblib")
ONNX_PATH = Path("models/wav2vec2_xlsr_int8.onnx")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("[VoiceShield] Loading Silero VAD & feature extractor...", flush=True)
vad_model = load_silero_vad()
extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
custom_clf = joblib.load(CUSTOM_CLF_PATH) if CUSTOM_CLF_PATH.exists() else None

ort_session = None
df_model = None
FAKE_IDX = 1

if ONNX_PATH.exists() and DEVICE == "cpu":
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(str(ONNX_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
    ENGINE_NAME = "ONNX INT8"
    print(f"[VoiceShield] Loaded fast {ENGINE_NAME} model from {ONNX_PATH}", flush=True)
else:
    from transformers import AutoModelForAudioClassification
    df_model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(DEVICE).eval()
    id2label = df_model.config.id2label
    FAKE_IDX = next((int(k) for k, v in id2label.items() if any(w in v.lower() for w in ("fake", "spoof"))), 1)
    ENGINE_NAME = f"PyTorch ({DEVICE.upper()})"
    print(f"[VoiceShield] Loaded {ENGINE_NAME} model", flush=True)

print(f"[VoiceShield] Custom trained classifier active: {custom_clf is not None}", flush=True)


@torch.inference_mode()
def extract_clean_speech(audio_16k: np.ndarray) -> tuple[np.ndarray | None, float]:
    wav_tensor = torch.from_numpy(audio_16k).float()
    speech_ts = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=16000)
    if not speech_ts:
        return None, 0.0
    speech_only = np.concatenate([audio_16k[t["start"]:t["end"]] for t in speech_ts])
    speech_ratio = len(speech_only) / max(1, len(audio_16k))
    if len(speech_only) < 8000:
        return None, round(speech_ratio, 2)
    speech_only = speech_only - float(np.mean(speech_only))
    peak = float(np.max(np.abs(speech_only)))
    if peak > 1e-4:
        speech_only = (speech_only / peak) * 0.9
    return speech_only.astype(np.float32), round(speech_ratio, 2)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


@torch.inference_mode()
def score_deepfake(speech_16k: np.ndarray) -> dict:
    if ort_session is not None:
        inputs = extractor(speech_16k, sampling_rate=16000, return_tensors="np")
        emb, logits = ort_session.run(["embedding", "logits"], {"input_values": inputs["input_values"].astype(np.float32)})
        w2v_emb = emb.squeeze()
        logits_np = logits[0]
    else:
        inputs = extractor(speech_16k, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out = df_model(**inputs, output_hidden_states=True)
        w2v_emb = out.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
        logits_np = out.logits[0].cpu().numpy()

    if custom_clf is not None:
        mfcc = librosa.feature.mfcc(y=speech_16k, sr=16000, n_mfcc=13)
        flat = librosa.feature.spectral_flatness(y=speech_16k)
        zcr = librosa.feature.zero_crossing_rate(y=speech_16k)
        feat = np.concatenate([
            w2v_emb, mfcc.mean(axis=1), mfcc.std(axis=1),
            [float(flat.mean()), float(flat.std()), float(zcr.mean()), float(zcr.std())]
        ]).reshape(1, -1)
        p_fake = float(custom_clf.predict_proba(feat)[0][1])
    else:
        p_fake = float(_softmax(logits_np)[FAKE_IDX])

    return {"p_fake": round(p_fake, 4), "engine": ENGINE_NAME}