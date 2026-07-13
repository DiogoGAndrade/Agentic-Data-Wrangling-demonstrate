import time
import requests
from typing import Optional, Dict, Any


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b-instruct",
        timeout: int = 2400,         # 40 min - large models on diabetes need >15 min
        max_retries: int = 3,
        backoff_sec: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec

    def list_models(self, timeout: int = 3) -> list:
        """Return the tags of models actually installed in this Ollama server.

        Calls GET /api/tags. Returns [] if the server is unreachable, so the UI
        can fall back to the static list and warn the user instead of crashing.
        """
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            return []

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        num_predict: int = 2048,      # cap generation length; reduced from 4096.
                                       # A valid JSON plan is ~800 tokens; 2048 is more
                                       # than sufficient and prevents runaway generation
                                       # that causes extreme latency on weaker models.
    ) -> str:
        url = f"{self.base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if system:
            payload["system"] = system

        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json().get("response", "")
            except Exception as e:
                last_err = e
                time.sleep(self.backoff_sec * attempt)

        raise RuntimeError(f"Ollama generate failed after {self.max_retries} retries. Last error: {last_err}")
