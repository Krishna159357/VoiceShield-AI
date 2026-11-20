import asyncio
from collections import deque
from pathlib import Path
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from backend.deepfake_model import extract_clean_speech, score_deepfake
from backend.speaker_verify import enroll_speaker, verify_speaker, list_speakers, delete_speaker
from backend.risk_engine import RiskEngine
from backend.audit_ledger import log_event, get_recent_events, verify_chain
from backend.mfa_alerts import send_mfa_alert

app = FastAPI()
SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE * 2

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
    print(f"[ENROLL] {result}", flush=True)
    return {**result, "speakers": list_speakers()}

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

            # Run Deepfake AI & Speaker Verification in parallel on the exact same clean speech
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