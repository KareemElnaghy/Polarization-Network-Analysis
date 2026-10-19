"""Thin client wrapper around a local Ollama HTTP API."""

from ollama import Client


class OllamaClient:
    """Small shared wrapper used by both polarization pipelines."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._client = Client(host=base_url)

    def generate(self, prompt: str, model: str) -> str:
        """Generate a response from ``model`` for ``prompt``."""
        response = self._client.generate(model=model, prompt=prompt)
        return response["response"]


_default_client = OllamaClient()


def generate(prompt: str, model: str) -> str:
    """Generate text through the shared local Ollama client."""
    return _default_client.generate(prompt, model)
