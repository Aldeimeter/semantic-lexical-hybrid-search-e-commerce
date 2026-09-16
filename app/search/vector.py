import time
from typing import TYPE_CHECKING

import httpx
import numpy as np

if TYPE_CHECKING:
    from .catalog import Product

# CPU/ONNX-backend TEI-образа режет батч на max_batch_requests=8
# (см. ARCHITECTURE.md п.2 — узнали из логов при верификации).
_MAX_BATCH = 8


class VectorSearcher:
    """HTTP-клиент к сервису text-embeddings-inference (TEI), поднятому
    отдельным сервисом в docker-compose (модель intfloat/multilingual-e5-base,
    веса в отдельном именованном volume, не в образе приложения).
    Никакой модели в этом процессе — только сетевой вызов."""

    def __init__(self, base_url: str, articles: list[str], matrix: np.ndarray, client: httpx.Client):
        self._base_url = base_url
        self._articles = articles
        self._matrix = matrix  # L2-нормализованная (n, dim)
        self._client = client

    @classmethod
    def build(
        cls,
        base_url: str,
        products: list["Product"],
        client: httpx.Client | None = None,
        ready_timeout: float = 180.0,
    ) -> "VectorSearcher":
        """base_url, например http://embeddings:80 (из docker-compose).
        Ждёт готовности сервиса (poll /health), затем кодирует все
        Наименования каталога с префиксом 'passage: '."""
        client = client or httpx.Client(timeout=60.0)
        cls._wait_until_ready(client, base_url, ready_timeout)

        articles = [p.article for p in products]
        texts = ["passage: " + p.search_text for p in products]
        vectors = cls._embed_batch(client, base_url, texts)
        matrix = cls._normalize(np.array(vectors, dtype=np.float32))
        return cls(base_url, articles, matrix, client)

    @staticmethod
    def _wait_until_ready(client: httpx.Client, base_url: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                resp = client.get(f"{base_url}/health", timeout=5.0)
                if resp.status_code == 200:
                    return
            except httpx.HTTPError as exc:
                last_error = exc
            time.sleep(2.0)
        raise RuntimeError(
            f"Embeddings service at {base_url} не стал готов за {timeout}с"
        ) from last_error

    @staticmethod
    def _embed_batch(client: httpx.Client, base_url: str, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _MAX_BATCH):
            chunk = texts[i : i + _MAX_BATCH]
            resp = client.post(f"{base_url}/embed", json={"inputs": chunk})
            resp.raise_for_status()
            out.extend(resp.json())
        return out

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def search(self, raw_query_text: str) -> dict[str, tuple[int, float]]:
        """article -> (rank, cosine_score), ранжируем ВЕСЬ каталог.
        raw_query_text — исходный текст запроса, БЕЗ лемматизации и
        БЕЗ fuzzy-коррекции, но с префиксом 'query: ' перед кодированием.

        rank начинается с 1, не с 0."""
        vector = self._embed_batch(self._client, self._base_url, ["query: " + raw_query_text])[0]
        query = self._normalize(np.array([vector], dtype=np.float32))[0]
        similarities = self._matrix @ query  # оба нормализованы -> косинус
        order = np.argsort(-similarities)
        return {
            self._articles[idx]: (rank + 1, float(similarities[idx]))
            for rank, idx in enumerate(order)
        }
