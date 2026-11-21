import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("voiceshield_audit.db")
GENESIS_HASH = "0" * 64

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("""
        CREATE TABLE IF NOT EXISTS audit_blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            speaker_id TEXT NOT NULL,
            p_fake REAL,
            speaker_match REAL,
            risk_score REAL NOT NULL,
            action TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            record_hash TEXT NOT NULL
        )
    """)
    return c

def _compute_hash(prev_hash: str, ts: str, sid: str, p_fake, s_match, risk: float, action: str) -> str:
    payload = f"{prev_hash}|{ts}|{sid}|{p_fake}|{s_match}|{risk}|{action}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def log_event(speaker_id: str, p_fake, speaker_match, risk_score: float, action: str) -> dict:
    with _conn() as c:
        row = c.execute("SELECT record_hash FROM audit_blocks ORDER BY id DESC LIMIT 1").fetchone()
        prev_hash = row["record_hash"] if row else GENESIS_HASH
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rec_hash = _compute_hash(prev_hash, ts, speaker_id, p_fake, speaker_match, risk_score, action)
        cur = c.execute(
            """INSERT INTO audit_blocks
               (timestamp, speaker_id, p_fake, speaker_match, risk_score, action, prev_hash, record_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, speaker_id, p_fake, speaker_match, risk_score, action, prev_hash, rec_hash),
        )
        c.commit()
        return {"block_id": cur.lastrowid, "timestamp": ts, "action": action, "record_hash": rec_hash[:16]}

def get_recent_events(limit: int = 12) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM audit_blocks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

def verify_chain() -> dict:
    with _conn() as c:
        rows = c.execute("SELECT * FROM audit_blocks ORDER BY id ASC").fetchall()
    expected_prev = GENESIS_HASH
    for r in rows:
        recomputed = _compute_hash(
            r["prev_hash"], r["timestamp"], r["speaker_id"],
            r["p_fake"], r["speaker_match"], r["risk_score"], r["action"]
        )
        if r["prev_hash"] != expected_prev or r["record_hash"] != recomputed:
            return {"valid": False, "total_blocks": len(rows), "tampered_block_id": r["id"]}
        expected_prev = r["record_hash"]
    return {"valid": True, "total_blocks": len(rows), "tampered_block_id": None}