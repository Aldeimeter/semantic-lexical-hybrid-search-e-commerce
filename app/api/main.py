import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

import build_enrichment
from search.catalog import load_catalog
from search.pipeline import SearchPipeline, SearchResult, build_pipeline

_pipeline: SearchPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    # products.csv монтируется как volume (не запечён в образ) и меняется
    # при каждой выгрузке из 1С — регенерируем enrichment.csv из ТЕКУЩЕГО
    # каталога на каждом старте, а не требуем ручного шага перед запуском
    # (см. ARCHITECTURE.md п.9 — процесс сопровождения полу-ручной на уровне
    # правил KEYWORD_OVERRIDES, но само применение правил к каталогу должно
    # быть автоматическим).
    build_enrichment.main()
    products = load_catalog("data/products.csv")
    embeddings_url = os.environ.get("EMBEDDINGS_URL", "http://embeddings:80")
    _pipeline = build_pipeline(
        products, embeddings_url, rrf_k=3, enrichment_path="data/enrichment.csv"
    )  # rrf_k: см. ARCHITECTURE.md — свип rrf_k
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Не просто "процесс жив" — проверяет, что build_pipeline() в lifespan
    реально завершился (каталог загружен, эмбеддинги от TEI получены).
    Если TEI ответил как healthy, но что-то в build_pipeline упало —
    процесс жив, но _pipeline=None, и health должен это показать, не 200."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not ready")
    return {"status": "ok"}


@app.get("/search")
def search(query: str, k: int = Query(default=3, ge=1)) -> list[SearchResult]:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not ready")
    try:
        return _pipeline.search(query, top_k=k, use_semantic=True)
    except Exception as exc:  # TEI недоступен в момент запроса и т.п.
        raise HTTPException(status_code=503, detail="search backend unavailable") from exc
