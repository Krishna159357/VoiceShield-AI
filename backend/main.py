import asyncio
from collections import deque
from pathlib import Path
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from backend.deepfake_model import analyze_window

app = FastAPI()
SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE * 2  # 2.0-second sliding window

@app.websocket("/ws/audio")
async def audio_stream(ws: WebSocket):
    await ws.accept()
    ring_buffer = deque(maxlen=WINDOW_SAMPLES)
    chunks_since_eval = 0
    is_processing = False

    async def run_inference(window_np: np.ndarray):
        nonlocal is_processing
        try:
            result = await asyncio.to_thread(analyze_window, window_np)
            print(f"[AI] {result}", flush=True)
            await ws.send_json(result)
        except Exception as e:
            print(f"[AI ERROR] {e}", flush=True)
        finally:
            is_processing = False

    try:
        while True:
            raw_bytes = await ws.receive_bytes()
            pcm_float = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            ring_buffer.extend(pcm_float)
            chunks_since_eval += 1

            # Evaluate every 1.0s (10 chunks of 100ms) without piling up slow frames
            if len(ring_buffer) == WINDOW_SAMPLES and chunks_since_eval >= 10 and not is_processing:
                chunks_since_eval = 0
                is_processing = True
                window = np.array(ring_buffer, dtype=np.float32)
                asyncio.create_task(run_inference(window))
    except WebSocketDisconnect:
        print("Caller disconnected cleanly.", flush=True)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")