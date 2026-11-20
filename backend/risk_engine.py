from collections import deque

class RiskEngine:
    def __init__(self):
        self.history = deque(maxlen=3)

    def reset(self):
        self.history.clear()

    def score(self, p_fake: float, s_match: float | None) -> dict:
        if s_match is None:
            # No enrolled voice yet — calibrate around laptop mic baseline
            raw_risk = max(0.0, min(100.0, (p_fake - 0.70) / 0.28 * 100.0))
        else:
            # 1. Strong speaker match + normal mic p_fake (< 0.96) => Genuine Speaker (SAFE)
            if s_match >= 0.45 and p_fake < 0.96:
                raw_risk = max(5.0, (1.0 - s_match) * 35.0)
            # 2. High-confidence AI clone mimicking the enrolled speaker (p_fake >= 0.96)
            elif p_fake >= 0.96:
                raw_risk = 88.0
            # 3. Unmatched voice / Impostor (s_match < 0.35)
            elif s_match < 0.35:
                raw_risk = 85.0 if p_fake >= 0.80 else 65.0
            # 4. Borderline match (0.35 - 0.45) => Medium Risk / MFA
            else:
                raw_risk = 62.0

        self.history.append(raw_risk)
        smoothed = round(sum(self.history) / len(self.history), 1)

        # Require at least 2 speech windows before hard-blocking (window 1 triggers MFA first)
        if smoothed >= 80 and len(self.history) >= 2:
            tier, action = "HIGH", "BLOCK"
        elif smoothed >= 55:
            tier, action = "MEDIUM", "VERIFY_MFA"
        elif smoothed >= 30:
            tier, action = "LOW", "WARN"
        else:
            tier, action = "SAFE", "ALLOW"

        return {"risk_score": smoothed, "risk_tier": tier, "action": action}