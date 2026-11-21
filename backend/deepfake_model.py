from pathlib import Path
import joblib
import librosa
import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
CUSTOM_CLF_PATH = Path("models/custom_deepfake_clf.joblib")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[VoiceShield] Loading Silero VAD & {MODEL_ID} on {DEVICE}...", flush=True)
vad_model = load_silero_vad()
extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
df_model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(DEVICE).eval()

custom_clf = joblib.load(CUSTOM_CLF_PATH) if CUSTOM_CLF_PATH.exists() else None
print(f"[VoiceShield] Custom trained classifier active: {custom_clf is not None}", flush=True)

id2label = df_model.config.id2label
FAKE_IDX = next((int(k) for k, v in id2label.items() if any(w in v.lower() for w in ("fake", "spoof"))), 1)

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

@torch.inference_mode()
def score_deepfake(speech_16k: np.ndarray) -> dict:
    inputs = extractor(speech_16k, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    if custom_clf is not None:
        out = df_model(**inputs, output_hidden_states=True)
        w2v_emb = out.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
        mfcc = librosa.feature.mfcc(y=speech_16k, sr=16000, n_mfcc=13)
        flat = librosa.feature.spectral_flatness(y=speech_16k)
        zcr = librosa.feature.zero_crossing_rate(y=speech_16k)
        feat = np.concatenate([
            w2v_emb, mfcc.mean(axis=1), mfcc.std(axis=1),
            [float(flat.mean()), float(flat.std()), float(zcr.mean()), float(zcr.std())]
        ]).reshape(1, -1)
        p_fake = float(custom_clf.predict_proba(feat)[0][1])
    else:
        logits = df_model(**inputs).logits
        p_fake = float(torch.softmax(logits, dim=-1)[0][FAKE_IDX].item())

    return {"p_fake": round(p_fake, 4)}