from pathlib import Path
import av
import joblib
import librosa
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from backend.deepfake_model import DEVICE, df_model, extract_clean_speech, extractor

REAL_DIR = Path("dataset/real")
FAKE_DIR = Path("dataset/fake")
MODEL_OUT = Path("models/custom_deepfake_clf.joblib")
MODEL_OUT.parent.mkdir(exist_ok=True)

def load_audio_16k(path: Path) -> np.ndarray:
    try:
        audio, _ = librosa.load(str(path), sr=16000, mono=True)
        return audio.astype(np.float32)
    except Exception:
        container = av.open(str(path))
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray().flatten())
        container.close()
        return np.concatenate(chunks).astype(np.float32)

@torch.inference_mode()
def extract_features(speech_16k: np.ndarray) -> np.ndarray:
    inputs = extractor(speech_16k, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = df_model(**inputs, output_hidden_states=True)
    w2v_emb = out.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()

    mfcc = librosa.feature.mfcc(y=speech_16k, sr=16000, n_mfcc=13)
    flat = librosa.feature.spectral_flatness(y=speech_16k)
    zcr = librosa.feature.zero_crossing_rate(y=speech_16k)
    acoustic = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        [float(flat.mean()), float(flat.std()), float(zcr.mean()), float(zcr.std())]
    ])
    return np.concatenate([w2v_emb, acoustic]).astype(np.float32)

def load_folder(folder: Path, label: int):
    X, y = [], []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".m4a", ".ogg", ".flac")]
    for f in files:
        audio = load_audio_16k(f)
        clean, _ = extract_clean_speech(audio)
        arr = clean if clean is not None else audio

        win, step = 32000, 16000
        if len(arr) >= win:
            chunks = [arr[i:i + win] for i in range(0, len(arr) - win + 1, step)]
        else:
            chunks = [arr]

        for ch in chunks:
            X.append(extract_features(ch))
            y.append(label)
        print(f"Processed {f.name} -> {len(chunks)} chunks (label={label})", flush=True)
    return X, y

def main():
    X_real, y_real = load_folder(REAL_DIR, 0)  # 0 = REAL HUMAN
    X_fake, y_fake = load_folder(FAKE_DIR, 1)  # 1 = AI / FAKE

    if len(X_real) < 4 or len(X_fake) < 4:
        print("ERROR: Add at least 2-3 audio files to BOTH dataset/real and dataset/fake first!")
        return

    X = np.array(X_real + X_fake)
    y = np.array(y_real + y_fake)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"\n=== Custom Model Validation Accuracy: {acc * 100:.1f}% ===")
    print(classification_report(y_te, preds, target_names=["REAL", "AI_FAKE"]))

    joblib.dump(clf, MODEL_OUT)
    print(f"Saved trained classifier to {MODEL_OUT}")

if __name__ == "__main__":
    main()