from collections import deque
from pathlib import Path
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

app = FastAPI()
SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE * 2  # 2.0-second sliding window

@app.websocket("/ws/audio")
async def audio_stream(ws: WebSocket):
    await ws.accept()
    ring_buffer = deque(maxlen=WINDOW_SAMPLES)
    chunks_since_eval = 0

    try:
        while True:
            raw_bytes = await ws.receive_bytes()
            pcm_float = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            ring_buffer.extend(pcm_float)
            chunks_since_eval += 1

            # Evaluate every 0.5s (5 chunks of 100ms) once 2.0s buffer is full
            if len(ring_buffer) == WINDOW_SAMPLES and chunks_since_eval >= 5:
                chunks_since_eval = 0
                window = np.array(ring_buffer, dtype=np.float32)
                rms = float(np.sqrt(np.mean(window ** 2)))
                print(f"[LIVE 16kHz] window=2.0s | rms={rms:.4f}")
                await ws.send_json({
                    "buffer_ready": True,
                    "rms": round(rms, 4),
                    "status": "SPEECH_DETECTED" if rms > 0.01 else "SILENCE"
                })
    except WebSocketDisconnect:
        print("Caller disconnected cleanly.")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")