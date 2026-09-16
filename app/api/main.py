import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

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


@app.get("/search")
def search(query: str, k: int = 3) -> list[SearchResult]:
    return _pipeline.search(query, top_k=k, use_semantic=True)
