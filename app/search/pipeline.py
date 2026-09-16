from dataclasses import dataclass

from .catalog import Product, apply_enrichment, load_enrichment
from .fuzzy import FuzzyCorrector, FuzzyDictionary
from .fusion import RankedList, rrf_merge
from .lexical import LexicalSearcher
from .normalize import lemmatize, lemmatize_catalog, tokenize
from .vector import VectorSearcher


@dataclass
class SearchResult:
    article: str
    name: str
    category: str
    price: float
    score: float


@dataclass
class SearchDebugInfo:
    """Полный, НЕСРЕЗАННЫЙ результат одного поиска — единственное место,
    где реально считаются list_a/list_b/merged. И SearchPipeline.search(),
    и evaluate() идут через search_full(), не дублируют вызовы lexical/vector/
    rrf_merge каждый по-своему."""

    list_a: RankedList
    list_b: RankedList | None
    merged: list[tuple[str, float]]


class SearchPipeline:
    def __init__(
        self,
        products: list[Product],
        fuzzy: FuzzyCorrector,
        lexical: LexicalSearcher,
        vector: VectorSearcher | None,
        rrf_k: float = 0,
    ):
        self._products_by_article = {p.article: p for p in products}
        self._fuzzy = fuzzy
        self._lexical = lexical
        self._vector = vector
        self._rrf_k = rrf_k

    def search_full(self, query: str, use_semantic: bool = True) -> SearchDebugInfo:
        """
        1. tokenize(query) -> lemmatize -> fuzzy.correct(lemma) для КАЖДОЙ
           леммы без условий (OOV-проверка и решение о коррекции — внутри
           FuzzyCorrector.correct())
        2. lexical.search(исправленные леммы) -> list_a
        3. если use_semantic и vector задан: vector.search(query — ИСХОДНЫЙ
           текст) -> list_b; иначе list_b = None (бейзлайн-аблация)
        4. rrf_merge(list_a, list_b, rrf_k) -> merged (НЕ обрезан)
        """
        tokens = tokenize(query)
        lemmas = lemmatize(tokens)
        corrected_lemmas = [self._fuzzy.correct(lemma) for lemma in lemmas]

        list_a = self._lexical.search(corrected_lemmas)
        list_b = self._vector.search(query) if (use_semantic and self._vector is not None) else None

        merged = rrf_merge(list_a, list_b, self._rrf_k)
        return SearchDebugInfo(list_a=list_a, list_b=list_b, merged=merged)

    def search(self, query: str, top_k: int = 3, use_semantic: bool = True) -> list[SearchResult]:
        """Тонкая обёртка: search_full(query, use_semantic).merged[:top_k],
        обогащение article -> Product (name, category, price) + score."""
        debug = self.search_full(query, use_semantic)
        results = []
        for article, score in debug.merged[:top_k]:
            product = self._products_by_article[article]
            results.append(
                SearchResult(
                    article=product.article,
                    name=product.name,
                    category=product.category,
                    price=product.price,
                    score=score,
                )
            )
        return results


def build_pipeline(
    products: list[Product],
    embeddings_base_url: str | None,
    rrf_k: float = 0,
    enrichment_path: str | None = None,
) -> SearchPipeline:
    """Единая точка сборки — вызывается и из evaluate.py:main(), и из
    api/main.py при старте. Лемматизирует каталог ОДИН раз и раздаёт
    результат обоим потребителям (FuzzyDictionary, LexicalSearcher).

    enrichment_path — опциональный отдельный файл (Артикул;Сценарий),
    джойнится к products по Артикулу ДО индексации (products.csv выгружается
    из 1С заново каждый день и не редактируется нами — обогащение живёт
    отдельно, см. Product.search_text в catalog.py)."""
    if enrichment_path:
        apply_enrichment(products, load_enrichment(enrichment_path))

    product_lemmas = lemmatize_catalog(products)

    fuzzy_dict = FuzzyDictionary.build(product_lemmas)
    fuzzy = FuzzyCorrector(fuzzy_dict)

    lexical = LexicalSearcher.build(products, product_lemmas)

    vector = VectorSearcher.build(embeddings_base_url, products) if embeddings_base_url else None

    return SearchPipeline(products, fuzzy, lexical, vector, rrf_k)
