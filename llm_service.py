from __future__ import annotations

from typing import Any

import requests
from openai import OpenAI

from config import settings


class LLMService:
    """Provider wrapper for chat and embedding calls.

    Default provider is Ollama. Switch to OpenAI by setting:

        LLM_PROVIDER=openai
        OPENAI_API_KEY=your_key_here
    """

    def __init__(self) -> None:
        self.provider = settings.llm_provider
        self._openai_client: OpenAI | None = None

        if self.provider == "openai":
            self._openai_client = OpenAI(api_key=settings.openai_api_key)

    def embed_text(self, text: str) -> list[float]:
        if self.provider == "ollama":
            return self._embed_with_ollama(text)

        return self._embed_with_openai(text)

    def generate_answer(self, question: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document-grounded RAG assistant for an AI Security course. "
                    "Answer only using the retrieved context. If the context does not contain "
                    "enough information, say that you do not know based on the provided documents. "
                    "Keep the answer clear, beginner-friendly, and concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Retrieved context:\n"
                    f"{context}\n\n"
                    "Question:\n"
                    f"{question}\n\n"
                    "Answer:"
                ),
            },
        ]

        if self.provider == "ollama":
            return self._chat_with_ollama(messages)

        return self._chat_with_openai(messages)

    def _embed_with_ollama(self, text: str) -> list[float]:
        url = f"{settings.ollama_base_url}/api/embeddings"
        payload = {"model": settings.ollama_embedding_model, "prompt": text}

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError(
                "Ollama did not return an embedding. Make sure the embedding model is pulled, "
                f"for example: ollama pull {settings.ollama_embedding_model}"
            )

        return [float(value) for value in embedding]

    def _chat_with_ollama(self, messages: list[dict[str, str]]) -> str:
        url = f"{settings.ollama_base_url}/api/chat"
        payload = {
            "model": settings.ollama_chat_model,
            "messages": messages,
            "stream": False,
        }

        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise RuntimeError(
                "Ollama did not return a chat response. Make sure the chat model is pulled, "
                f"for example: ollama pull {settings.ollama_chat_model}"
            )

        return content

    def _embed_with_openai(self, text: str) -> list[float]:
        if self._openai_client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        response = self._openai_client.embeddings.create(
            model=settings.openai_embedding_model,
            input=text,
        )
        return [float(value) for value in response.data[0].embedding]

    def _chat_with_openai(self, messages: list[dict[str, str]]) -> str:
        if self._openai_client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        response = self._openai_client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=messages,
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()
