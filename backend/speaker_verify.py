import re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from backend.deepfake_model import extract_clean_speech

ENROLL_DIR = Path("enrolled_voices")
ENROLL_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[VoiceShield] Loading SpeechBrain ECAPA-TDNN on {DEVICE}...", flush=True)
spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/ecapa",
    run_opts={"device": DEVICE},
    local_strategy=LocalStrategy.COPY,
)

def _clean_id(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_").lower()
    return cleaned or "speaker"

@torch.inference_mode()
def extract_embedding(speech_16k: np.ndarray) -> np.ndarray:
    wav = torch.from_numpy(speech_16k).float().unsqueeze(0).to(DEVICE)
    emb = spk_model.encode_batch(wav).squeeze()
    emb = F.normalize(emb, p=2, dim=0)
    return emb.cpu().numpy().astype(np.float32)

def enroll_speaker(audio_16k: np.ndarray, name: str) -> dict:
    sid = _clean_id(name)
    clean_speech, _ = extract_clean_speech(audio_16k)
    audio_use = clean_speech if clean_speech is not None else audio_16k

    # Split into 2s overlapping chunks and average into a robust voiceprint centroid
    chunk_len = 32000
    step = 16000
    if len(audio_use) > chunk_len:
        embs = [extract_embedding(audio_use[i:i + chunk_len]) for i in range(0, len(audio_use) - chunk_len + 1, step)]
        centroid = np.mean(embs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    else:
        centroid = extract_embedding(audio_use)

    np.save(ENROLL_DIR / f"{sid}.npy", centroid.astype(np.float32))
    return {"enrolled": True, "speaker_id": sid, "display_name": name.strip(), "embedding_dim": int(centroid.shape[0])}

def list_speakers() -> list[str]:
    return sorted(p.stem for p in ENROLL_DIR.glob("*.npy"))

def delete_speaker(speaker_id: str) -> bool:
    path = ENROLL_DIR / f"{_clean_id(speaker_id)}.npy"
    if path.exists():
        path.unlink()
        return True
    return False

def verify_speaker(speech_16k: np.ndarray, speaker_id: str) -> float | None:
    path = ENROLL_DIR / f"{_clean_id(speaker_id)}.npy"
    if not path.exists():
        return None
    enrolled = torch.from_numpy(np.load(path))
    live = torch.from_numpy(extract_embedding(speech_16k))
    sim = F.cosine_similarity(live, enrolled, dim=0).item()
    return round(float(sim), 4)