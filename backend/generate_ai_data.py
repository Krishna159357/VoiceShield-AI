import asyncio
from pathlib import Path
import edge_tts

OUT_DIR = Path("dataset/fake")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    ("en-IN-PrabhatNeural", "Hello, this is an urgent call from the chief executive office. Please approve the wire transfer immediately."),
    ("en-IN-NeerjaNeural", "Good morning, I am calling from the finance department. We need you to verify the account details right now."),
    ("en-US-GuyNeural", "Hi, I am stuck in a meeting and need you to process this confidential payment before five o'clock today."),
    ("en-US-JennyNeural", "Please confirm the one time password sent to your phone so we can unlock your corporate banking access."),
    ("en-GB-RyanNeural", "This is an emergency authorization request. Do not discuss this call with anyone else in the office."),
    ("en-IN-PrabhatNeural", "Our server audit found an immediate security breach. Transfer the reserve funds to the safe account now."),
    ("en-IN-NeerjaNeural", "Hello team, please update the vendor bank account number and release the invoice amount within ten minutes."),
    ("en-US-ChristopherNeural", "I need a quick favor. Can you send the authorization code over this call? I will email the paperwork later."),
]

async def main():
    for i, (voice, text) in enumerate(SAMPLES, 1):
        out_file = OUT_DIR / f"ai_sample_{i}.mp3"
        await edge_tts.Communicate(text, voice).save(str(out_file))
        print(f"Saved {out_file}", flush=True)

asyncio.run(main())