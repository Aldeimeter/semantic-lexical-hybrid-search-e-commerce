import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from search.catalog import load_catalog
from search.pipeline import SearchPipeline, SearchResult, build_pipeline

_pipeline: SearchPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    products = load_catalog("data/products.csv")
    embeddings_url = os.environ.get("EMBEDDINGS_URL", "http://embeddings:80")
    _pipeline = build_pipeline(products, embeddings_url, rrf_k=3)  # см. ARCHITECTURE.md — свип rrf_k
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
def search(query: str, k: int = 3) -> list[SearchResult]:
    return _pipeline.search(query, top_k=k, use_semantic=True)
