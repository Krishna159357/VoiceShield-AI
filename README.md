# VoiceShield AI — Team SHRESTH
# VoiceShield AI

**Catching cloned voices while the call is still happening.**

Built by team SHRESTH (Yash, Shrwan, Krishna Kant, Sujit) for Hack-N-Pitch at E-Summit '26, Jadavpur University. Track: Blockchain & Cybersecurity.

## Why we built this

Voice-cloning scams usually get caught after the money is gone. A cloned voice calls, sounds exactly like your boss or your mother, and the transfer goes through before anyone doubts it.

We wanted to flip that. VoiceShield listens to the call as it happens, asks two questions (*Is this voice synthetic? Is it actually the person it claims to be?*), and steps in before the damage is done. We call the approach **Detect → Decide → Prevent**.

## How it works

Audio comes in at 16 kHz, either from the browser mic or an uploaded file. We cut it into 3-second windows that slide forward 1 second at a time, and a Silero VAD gate skips the silent parts. Each window goes through two models in parallel:

- **Deepfake detector:** Wav2Vec2 XLS-R embeddings feed a classifier we trained ourselves to tell real speech from AI speech.
- **Speaker verification:** SpeechBrain's ECAPA-TDNN compares the voice to the enrolled user with cosine similarity.

A risk engine combines both signals into a smoothed score from 0 to 100, and that score decides what happens next:

| Risk | Action |
|------|--------|
| Safe | Allow the call |
| Low | Warn the user |
| Medium | Send a 6-digit OTP over Telegram for verification |
| High | Cut the call automatically |

Every decision is logged to a tamper-evident audit ledger. The ledger stores metadata only, never audio.

```text
Live 16 kHz audio (browser AudioWorklet / uploaded file)
        │
        ▼
Ring buffer (3.0s window, 1.0s hop) + Silero VAD
        │
        ├─────────────────────────┐
        ▼                         ▼
Wav2Vec2 XLS-R +           SpeechBrain ECAPA-TDNN
custom classifier          (192-dim speaker embedding,
(AI vs. real)               cosine match vs. enrolled user)
        │                         │
        └────────────┬────────────┘
                     ▼
         Smoothed risk engine (0–100)
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
  Low: warn    Medium: Telegram   High: block call
               OTP check
                     │
                     ▼
     SHA-256 hash-chained audit ledger (SQLite)
```

## What it can do

**Live call protection.** The browser captures raw 16 kHz PCM16 audio through an AudioWorklet and streams it over FastAPI WebSockets. You get a live waveform while both models run in parallel.

**A deepfake detector we trained ourselves (95.0% validation accuracy).** We took hidden-state embeddings from a frozen, multilingual `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` model, added MFCCs, spectral flatness and zero-crossing rate, and trained a classifier on real microphone recordings and neural TTS samples. On validation: REAL recall 1.00, AI_FAKE precision 1.00, F1 0.95.

**Voiceprint enrollment that doesn't keep your voice.** Enroll by name with a 5-second live recording or an uploaded file. We use `speechbrain/spkrec-ecapa-voxceleb` to produce a 192-dimensional embedding, save only that `.npy` vector, and discard the raw audio immediately.

**Graded prevention instead of a single on/off switch.** The response escalates from allow to warn to OTP verification to blocking the call.

**Forensic scanner.** Upload a `.wav`, `.mp3` or `.m4a` file, or record a short clip, and it tells you whether the speech sounds synthetic or human and which enrolled speaker it's closest to.

**Tamper-evident audit ledger.** Each entry holds only a timestamp, speaker ID, `p_fake`, speaker match, risk score and action. Entries are chained with `SHA-256(prev_hash | payload)`, so editing any past record breaks every hash after it. One click hits `/api/audit/verify` to check the whole chain.

## Tech stack

| Part | What we used |
|------|--------------|
| Backend & streaming | FastAPI, Uvicorn, WebSockets, Python 3.11 |
| Voice activity detection | Silero VAD |
| Deepfake detection | Wav2Vec2-Large-XLSR + scikit-learn classifier head |
| Speaker verification | SpeechBrain ECAPA-TDNN |
| Frontend & capture | HTML5, CSS3, vanilla JS, Web Audio AudioWorklet |
| MFA alerts & ledger | Telegram Bot API, SQLite + SHA-256 hash chain |

## Getting started

**1. Clone and install**

```bash
git clone https://github.com/Krishna159357/VoiceShield-AI.git
cd VoiceShield-AI
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# Mac / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

**2. Set up Telegram alerts (optional)**

Create a `.env` file in the project root (there's a template in `.env.example`):

```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

**3. Train your own classifier (optional)**

Put real human voice clips in `dataset/real/` and AI-generated clips in `dataset/fake/`. Or let the script generate neural TTS samples for you:

```bash
python -m backend.generate_ai_data
python -m backend.train_model
```

**4. Run it**

```bash
python -m uvicorn backend.main:app --port 8000
```

Then open http://localhost:8000 in Chrome.

Demoing offline? Once the model weights are cached, set `HF_HUB_OFFLINE=1` before launching Uvicorn.

## Project layout

```text
voiceshield-ai/
├── backend/
│   ├── main.py               # FastAPI routes, WebSockets, enrollment & scanner
│   ├── deepfake_model.py     # Silero VAD + Wav2Vec2-XLSR + classifier inference
│   ├── speaker_verify.py     # ECAPA-TDNN enrollment & cosine matching
│   ├── risk_engine.py        # Dual-signal risk scoring, smoothed over 3 windows
│   ├── mfa_alerts.py         # Telegram Bot API OTP alerts
│   ├── audit_ledger.py       # Metadata-only SQLite + SHA-256 hash chain
│   ├── generate_ai_data.py   # Neural TTS training data generator (edge-tts)
│   └── train_model.py        # Classifier trainer & evaluator
├── frontend/
│   ├── index.html            # Dashboard, enrollment, scanner & ledger view
│   └── pcm-processor.js      # 16 kHz Int16 PCM AudioWorklet processor
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```
