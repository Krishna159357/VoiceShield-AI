from collections import deque
import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ASVspoof 2021-trained model (much better on real laptop microphones & room acoustics)
MODEL_ID = "Vansh180/deepfake-audio-wav2vec2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[VoiceShield] Loading Silero VAD & {MODEL_ID} on {DEVICE}...", flush=True)
vad_model = load_silero_vad()
extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
df_model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(DEVICE).eval()

id2label = df_model.config.id2label
print(f"[VoiceShield] Model labels: {id2label}", flush=True)
FAKE_IDX = next(
    (int(k) for k, v in id2label.items() if any(w in v.lower() for w in ("fake", "spoof", "label_1"))),
    1,
)

# Smooth across the last 3 speech windows to stop flickering
recent_scores = deque(maxlen=3)

@torch.inference_mode()
def analyze_window(audio_16k: np.ndarray) -> dict:
    # 1. Silero VAD — extract ONLY active speech frames (strip room silence)
    wav_tensor = torch.from_numpy(audio_16k)
    speech_ts = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=16000)
    if not speech_ts:
        return {"is_speech": False, "speech_ratio": 0.0, "p_fake": None, "verdict": "SILENCE"}

    speech_chunks = [audio_16k[t["start"]:t["end"]] for t in speech_ts]
    speech_only = np.concatenate(speech_chunks)
    speech_ratio = len(speech_only) / len(audio_16k)

    # Need at least 0.5s of actual speech before scoring
    if len(speech_only) < 8000:
        return {"is_speech": False, "speech_ratio": round(speech_ratio, 2), "p_fake": None, "verdict": "SILENCE"}

    # 2. Peak-normalize mic volume so quiet speech isn't misread as noise
    peak = float(np.max(np.abs(speech_only)))
    if peak > 1e-4:
        speech_only = speech_only / peak * 0.9

    # 3. Wav2Vec2 Deepfake inference on clean normalized speech
    inputs = extractor(speech_only, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = df_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    raw_fake = float(probs[FAKE_IDX].item())

    # 4. 3-window moving average for smooth, stable output
    recent_scores.append(raw_fake)
    smoothed_fake = float(sum(recent_scores) / len(recent_scores))

    verdict = "HIGH_RISK_FAKE" if smoothed_fake >= 0.75 else ("SUSPICIOUS" if smoothed_fake >= 0.45 else "LIKELY_REAL")
    return {
        "is_speech": True,
        "speech_ratio": round(speech_ratio, 2),
        "p_fake": round(smoothed_fake, 4),
        "raw_p_fake": round(raw_fake, 4),
        "verdict": verdict,
    }