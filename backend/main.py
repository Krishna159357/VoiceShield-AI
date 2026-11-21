import asyncio
import tempfile
from collections import deque
from pathlib import Path
import librosa
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from backend.deepfake_model import extract_clean_speech, score_deepfake
from backend.speaker_verify import enroll_speaker, verify_speaker, list_speakers, delete_speaker
from backend.risk_engine import RiskEngine
from backend.audit_ledger import log_event, get_recent_events, verify_chain
from backend.mfa_alerts import send_mfa_alert

app = FastAPI()
SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE * 3

def _load_uploaded_audio(raw_bytes: bytes, suffix: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".wav") as tmp:
        tmp.write(raw_bytes)
        tmp_path = Path(tmp.name)
    try:
        audio_16k, _ = librosa.load(str(tmp_path), sr=SAMPLE_RATE, mono=True)
        return audio_16k.astype(np.float32)
    finally:
        tmp_path.unlink(missing_ok=True)  # Privacy-first: delete raw file immediately

def _analyze_audio_file(audio_16k: np.ndarray) -> dict:
    duration = round(len(audio_16k) / SAMPLE_RATE, 2)
    clean_speech, speech_ratio = extract_clean_speech(audio_16k)
    audio_use = clean_speech if clean_speech is not None else audio_16k

    # Score across overlapping 2s windows for stable forensic accuracy
    chunk_len, step = 32000, 16000
    if len(audio_use) > chunk_len:
        scores = [
            score_deepfake(audio_use[i:i + chunk_len])["p_fake"]
            for i in range(0, len(audio_use) - chunk_len + 1, step)
        ]
    else:
        scores = [score_deepfake(audio_use)["p_fake"]]

    avg_fake = round(float(np.mean(scores)), 4)
    peak_fake = round(float(np.max(scores)), 4)

    # Check against all enrolled speakers to spot impersonation
    best_speaker, best_match = None, -1.0
    for sid in list_speakers():
        sim = verify_speaker(audio_use[:min(len(audio_use), 80000)], sid)
        if sim is not None and sim > best_match:
            best_speaker, best_match = sid, sim

    if avg_fake >= 0.65:
        verdict = "AI / CLONED VOICE"
        risk_score = round(avg_fake * 100, 1)
        log_event(best_speaker or "unknown_file", avg_fake, best_match if best_speaker else None, risk_score, "FILE_AI_DETECTED")
    elif avg_fake >= 0.45:
        verdict = "SUSPICIOUS / UNCERTAIN"
        risk_score = round(avg_fake * 100, 1)
    else:
        verdict = "REAL HUMAN VOICE"
        risk_score = round(avg_fake * 100, 1)

    return {
        "verdict": verdict,
        "p_fake_avg": avg_fake,
        "p_fake_peak": peak_fake,
        "risk_score": risk_score,
        "duration_sec": duration,
        "speech_ratio": speech_ratio,
        "closest_speaker": best_speaker,
        "speaker_match": round(best_match, 4) if best_speaker else None,
    }

@app.get("/api/speakers")
async def get_speakers():
    return {"speakers": list_speakers()}

@app.delete("/api/speakers/{speaker_id}")
async def remove_speaker(speaker_id: str):
    return {"deleted": delete_speaker(speaker_id), "speakers": list_speakers()}

@app.post("/api/enroll")
async def enroll_endpoint(request: Request, name: str = "ceo"):
    raw_bytes = await request.body()
    pcm_float = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    result = await asyncio.to_thread(enroll_speaker, pcm_float, name)
    return {**result, "speakers": list_speakers()}

@app.post("/api/enroll-file")
async def enroll_file_endpoint(name: str = Form(...), file: UploadFile = File(...)):
    raw_bytes = await file.read()
    suffix = Path(file.filename or "audio.wav").suffix
    audio_16k = await asyncio.to_thread(_load_uploaded_audio, raw_bytes, suffix)
    result = await asyncio.to_thread(enroll_speaker, audio_16k, name)
    print(f"[ENROLL FILE] {result}", flush=True)
    return {**result, "speakers": list_speakers()}

@app.post("/api/analyze-file")
async def analyze_file_endpoint(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    suffix = Path(file.filename or "audio.wav").suffix
    audio_16k = await asyncio.to_thread(_load_uploaded_audio, raw_bytes, suffix)
    result = await asyncio.to_thread(_analyze_audio_file, audio_16k)
    print(f"[FILE SCAN] {result}", flush=True)
    return result
@app.post("/api/analyze-pcm")
async def analyze_pcm_endpoint(request: Request):
    raw_bytes = await request.body()
    pcm_float = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    result = await asyncio.to_thread(_analyze_audio_file, pcm_float)
    print(f"[REC SCAN] {result}", flush=True)
    return result

@app.get("/api/audit")
async def audit_list():
    return {"events": get_recent_events(12), "integrity": verify_chain()}

@app.get("/api/audit/verify")
async def audit_verify():
    return verify_chain()

@app.websocket("/ws/audio")
async def audio_stream(ws: WebSocket, speaker_id: str = "ceo"):
    await ws.accept()
    ring_buffer = deque(maxlen=WINDOW_SAMPLES)
    risk_engine = RiskEngine()
    chunks_since_eval = 0
    is_processing = False

    async def run_pipeline(window_np: np.ndarray):
        nonlocal is_processing
        try:
            clean_speech, speech_ratio = await asyncio.to_thread(extract_clean_speech, window_np)
            if clean_speech is None:
                await ws.send_json({
                    "is_speech": False, "speech_ratio": speech_ratio, "p_fake": None,
                    "speaker_id": speaker_id, "speaker_match": None,
                    "risk_score": 0, "risk_tier": "SILENCE", "action": "WAIT"
                })
                return

            df_res, s_match = await asyncio.gather(
                asyncio.to_thread(score_deepfake, clean_speech),
                asyncio.to_thread(verify_speaker, clean_speech, speaker_id),
            )
            risk = risk_engine.score(df_res["p_fake"], s_match)
            payload = {
                "is_speech": True, "speech_ratio": speech_ratio,
                **df_res, "speaker_id": speaker_id, "speaker_match": s_match, **risk
            }

            if risk["action"] in ("WARN", "VERIFY_MFA", "BLOCK"):
                block = await asyncio.to_thread(
                    log_event, speaker_id, df_res["p_fake"], s_match, risk["risk_score"], risk["action"]
                )
                payload["audit_block"] = block

            if risk["action"] in ("VERIFY_MFA", "BLOCK"):
                mfa = await asyncio.to_thread(send_mfa_alert, speaker_id, risk["risk_score"], risk["action"])
                payload["mfa"] = mfa

            print(f"[PIPELINE] {payload}", flush=True)
            await ws.send_json(payload)
        except Exception as e:
            print(f"[PIPELINE ERROR] {e}", flush=True)
        finally:
            is_processing = False

    try:
        while True:
            raw_bytes = await ws.receive_bytes()
            pcm_float = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            ring_buffer.extend(pcm_float)
            chunks_since_eval += 1

            if len(ring_buffer) == WINDOW_SAMPLES and chunks_since_eval >= 10 and not is_processing:
                chunks_since_eval = 0
                is_processing = True
                window = np.array(ring_buffer, dtype=np.float32)
                asyncio.create_task(run_pipeline(window))
    except WebSocketDisconnect:
        print("Caller disconnected cleanly.", flush=True)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")