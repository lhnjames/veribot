"""Minimal DeepSeek chat client for the offline evidence bank.

The API key is read at runtime from a 0600 file OUTSIDE the repo (default
/path/to/data_root/secrets/deepseek_api_key) or the DEEPSEEK_API_KEY env var. It
is never written to source, argv, ps, logs, the manifest, raw request dumps, or any printed output;
this module deliberately has no code path that echoes it.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

DEFAULT_KEY_FILE = "/path/to/data_root/secrets/deepseek_api_key"
ENDPOINT = "https://api.deepseek.com/chat/completions"


def load_api_key(key_file: str = DEFAULT_KEY_FILE) -> str:
    env = os.environ.get("DEEPSEEK_API_KEY")
    if env:
        return env.strip()
    p = Path(key_file)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "no DeepSeek API key: set DEEPSEEK_API_KEY or place it in "
        f"{key_file} (0600). Never pass it on the command line."
    )


class DeepSeekClient:
    def __init__(self, model: str = "deepseek-v4-pro", key_file: str = DEFAULT_KEY_FILE,
                 timeout: int = 120):
        self._key = load_api_key(key_file)
        self.model = model
        self.timeout = timeout

    def chat(self, system: str, user: str, *, temperature: float = 0.2,
             max_tokens: int = 1024, response_json: bool = True) -> dict:
        """One call. Returns {content, usage, latency_s}. Raises on HTTP/parse error (caller retries)."""
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_json:
            body["response_format"] = {"type": "json_object"}
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(ENDPOINT, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self._key}")  # only place the key is used
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return {
            "content": payload["choices"][0]["message"]["content"],
            "usage": payload.get("usage", {}),
            "latency_s": round(time.time() - t0, 2),
        }
