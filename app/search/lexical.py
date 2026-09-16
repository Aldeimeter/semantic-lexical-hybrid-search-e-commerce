from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from .catalog import Product


class LexicalSearcher:
    """BM25 (rank_bm25) по леммам поля Наименование."""

    def __init__(self, articles: list[str], bm25: BM25Okapi):
        self._articles = articles
        self._bm25 = bm25

    @classmethod
    def build(cls, products: list["Product"], product_lemmas: list[list[str]]) -> "LexicalSearcher":
        """product_lemmas — результат lemmatize_catalog(products), параллельный
        products по индексу; ТОТ ЖЕ вызов, что передаётся в
        FuzzyDictionary.build() — не лемматизирует Наименование заново сам."""
        articles = [p.article for p in products]
        bm25 = BM25Okapi(product_lemmas)
        return cls(articles, bm25)

    def search(self, lemmas: list[str]) -> dict[str, tuple[int, float]]:
        """article -> (rank, bm25_score), ранжируем ВЕСЬ каталог.
        Товары с score == 0 (нет ни одного общего слова) НЕ включаются
        в результат — считаются отсутствующими, а не 'последними по рангу'.

        rank начинается с 1, не с 0 (иначе при rrf_k=0 — ZeroDivisionError)."""
        if not lemmas:
            return {}
        scores = self._bm25.get_scores(lemmas)
        pairs = [
            (self._articles[i], float(scores[i]))
            for i in range(len(scores))
            if scores[i] > 0
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return {article: (rank + 1, score) for rank, (article, score) in enumerate(pairs)}
