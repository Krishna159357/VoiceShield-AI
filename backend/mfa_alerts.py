import json
import os
import random
import time
import urllib.request
from pathlib import Path

_last_alert_time = 0.0

def _load_env():
    env_file = Path(".env")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()

def send_mfa_alert(speaker_id: str, risk_score: float, action: str) -> dict:
    global _last_alert_time
    now = time.time()
    if now - _last_alert_time < 20:
        return {"sent": False, "reason": "cooldown"}

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    otp = random.randint(100000, 999999)

    if not token or not chat_id:
        print(f"[MFA LOCAL] OTP={otp} | speaker={speaker_id} | action={action} (add .env for Telegram)", flush=True)
        _last_alert_time = now
        return {"sent": False, "otp": otp, "reason": "no_telegram_env"}

    text = (
        f"VoiceShield AI Security Alert\n"
        f"Claimed Caller: {speaker_id}\n"
        f"Risk Score: {risk_score}/100 ({action})\n"
        f"Verification OTP: {otp}\n"
        f"If this is NOT you, do not authorize any transfer."
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            _last_alert_time = now
            return {"sent": resp.status == 200, "otp": otp}
    except Exception as e:
        print(f"[MFA ERROR] {e}", flush=True)
        return {"sent": False, "otp": otp, "error": str(e)}