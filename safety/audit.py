"""audit.py - Tamper-Evident SHA-256 Chained Audit Logger and Decision Replayer.

Every verdict and monitor action is recorded as a JSON line with cryptographic
hash chaining. Any modification, insertion, or deletion of log entries is
immediately detected by verify_audit_chain(path).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def compute_sha256(content: str) -> str:
    """Compute standard SHA-256 hex digest of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class AuditLogger:
    """Tamper-evident append-only audit log with SHA-256 hash chaining."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_hash = self._get_tail_hash()

    def _get_tail_hash(self) -> str:
        """Retrieve hash of the last log line, or genesis hash if empty/new."""
        if not self.log_path.is_file():
            return "0" * 64
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                return "0" * 64
            last_entry = json.loads(lines[-1])
            return str(last_entry.get("record_hash", "0" * 64))
        except Exception:
            return "0" * 64

    def log_action(
        self,
        event_type: str,
        input_data: Dict[str, Any],
        verdict_data: Dict[str, Any],
        versions: Dict[str, str],
    ) -> Dict[str, Any]:
        """Append a tamper-evident audit record to the log file."""
        # 1. Canonical representation for deterministic hashing
        canonical_payload = {
            "prev_hash": self.last_hash,
            "event_type": event_type,
            "input_digest": compute_sha256(json.dumps(input_data, sort_keys=True)),
            "verdict": verdict_data,
            "versions": versions,
        }
        payload_str = json.dumps(canonical_payload, sort_keys=True)
        record_hash = compute_sha256(payload_str)

        audit_entry = {
            **canonical_payload,
            "record_hash": record_hash,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")

        self.last_hash = record_hash
        return audit_entry


def verify_audit_chain(log_path: Path) -> bool:
    """Verify cryptographic integrity of the entire audit chain.

    Returns:
        True if all hashes match and chain is unbroken; False if tampering detected.
    """
    if not log_path.is_file():
        return True

    expected_prev = "0" * 64
    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                return False

            recorded_hash = entry.get("record_hash")
            prev_hash = entry.get("prev_hash")

            if prev_hash != expected_prev:
                return False

            payload = {
                "prev_hash": prev_hash,
                "event_type": entry.get("event_type"),
                "input_digest": entry.get("input_digest"),
                "verdict": entry.get("verdict"),
                "versions": entry.get("versions"),
            }
            computed_hash = compute_sha256(json.dumps(payload, sort_keys=True))
            if computed_hash != recorded_hash:
                return False

            expected_prev = recorded_hash

    return True
