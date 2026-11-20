from collections import deque
import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "Vansh180/deepfake-audio-wav2vec2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[VoiceShield] Loading Silero VAD & {MODEL_ID} on {DEVICE}...", flush=True)
vad_model = load_silero_vad()
extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
df_model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(DEVICE).eval()

id2label = df_model.config.id2label
print(f"[VoiceShield] Model labels: {id2label}", flush=True)
FAKE_IDX = next(
    (int(k) for k, v in id2label.items() if any(w in v.lower() for w in ("fake", "spoof"))),
    1,
)

recent_scores = deque(maxlen=3)

@torch.inference_mode()
def extract_clean_speech(audio_16k: np.ndarray) -> tuple[np.ndarray | None, float]:
    wav_tensor = torch.from_numpy(audio_16k).float()
    speech_ts = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=16000)
    if not speech_ts:
        return None, 0.0

    speech_only = np.concatenate([audio_16k[t["start"]:t["end"]] for t in speech_ts])
    speech_ratio = len(speech_only) / max(1, len(audio_16k))
    if len(speech_only) < 9600:  # Need at least 0.6s of real speech
        return None, round(speech_ratio, 2)

    # Remove DC offset & peak-normalize cleanly
    speech_only = speech_only - float(np.mean(speech_only))
    peak = float(np.max(np.abs(speech_only)))
    if peak > 1e-4:
        speech_only = (speech_only / peak) * 0.9

    return speech_only.astype(np.float32), round(speech_ratio, 2)

@torch.inference_mode()
def score_deepfake(speech_16k: np.ndarray) -> dict:
    inputs = extractor(speech_16k, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = df_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    raw_fake = float(probs[FAKE_IDX].item())

    recent_scores.append(raw_fake)
    smoothed_fake = float(sum(recent_scores) / len(recent_scores))
    return {"p_fake": round(smoothed_fake, 4), "raw_p_fake": round(raw_fake, 4)}