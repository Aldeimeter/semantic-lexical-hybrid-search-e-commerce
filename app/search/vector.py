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
    ) -> "VectorSearcher":
        """base_url, например http://embeddings:80 (из docker-compose).
        Кодирует все Наименования каталога с префиксом 'passage: '.

        Готовность сервиса — забота docker-compose.yml (`depends_on:
        condition: service_healthy`), не этого кода: compose не запускает
        контейнер app, пока embeddings не станет healthy. Раньше здесь был
        свой poll-цикл с отдельным таймаутом (`ready_timeout`) — избыточный
        дубль той же гарантии, только с несинхронизированным вторым
        таймаутом (см. ARCHITECTURE.md). Единственное место, где нужен
        щедрый запас на медленную сеть/железо, — start_period в
        docker-compose.yml, не здесь."""
        client = client or httpx.Client(timeout=60.0)

        articles = [p.article for p in products]
        texts = ["passage: " + p.search_text for p in products]
        vectors = cls._embed_batch(client, base_url, texts)
        matrix = cls._normalize(np.array(vectors, dtype=np.float32))
        return cls(base_url, articles, matrix, client)

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
