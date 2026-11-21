from collections import deque

class RiskEngine:
    def __init__(self):
        self.history = deque(maxlen=3)

    def reset(self):
        self.history.clear()

    def score(self, p_fake: float, s_match: float | None) -> dict:
        if s_match is None:
            raw_risk = p_fake * 100.0
        else:
            # 1. AI Deepfake / Cloned Voice detected (even if speaker match is high!)
            if p_fake >= 0.72:
                raw_risk = 88.0
            # 2. Genuine enrolled owner (s_match >= 0.28 and real human voice)
            elif s_match >= 0.28 and p_fake < 0.60:
                raw_risk = max(5.0, (0.55 - min(s_match, 0.55)) * 40.0)
            # 3. Clear impostor / different human voice (s_match < 0.20)
            elif s_match < 0.20:
                raw_risk = 82.0
            # 4. Borderline / noisy speech => Medium or Warn
            else:
                raw_risk = 45.0 if p_fake < 0.60 else 65.0

        self.history.append(raw_risk)
        smoothed = round(sum(self.history) / len(self.history), 1)

        if smoothed >= 78 and len(self.history) >= 2:
            tier, action = "HIGH", "BLOCK"
        elif smoothed >= 55:
            tier, action = "MEDIUM", "VERIFY_MFA"
        elif smoothed >= 30:
            tier, action = "LOW", "WARN"
        else:
            tier, action = "SAFE", "ALLOW"

        return {"risk_score": smoothed, "risk_tier": tier, "action": action}