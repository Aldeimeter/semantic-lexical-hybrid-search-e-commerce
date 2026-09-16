RankedList = dict[str, tuple[int, float]]


def rrf_merge(
    list_a: RankedList,
    list_b: RankedList | None,
    rrf_k: float = 0,
) -> list[tuple[str, float]]:
    """RRF: score(article) = 1/(rrf_k + rank_a) + 1/(rrf_k + rank_b)
    (0, если отсутствует в списке). list_b=None -> бейзлайн Fuzzy->Lexical
    (только list_a, без слияния).

    Предполагает ранги с 1 (не с 0) в обоих входных списках — при rrf_k=0
    нулевой ранг даёт ZeroDivisionError на самом релевантном товаре.

    Сортировка результата: (score убыв., min(rank_a, rank_b) возр., article возр.)
    — при равном score предпочитаем товар с лучшим рангом хотя бы в одной
    ветке, НЕ товар, подтверждённый большим числом веток (см. ARCHITECTURE.md
    п.9 — почему это отклонено)."""
    articles = set(list_a) | (set(list_b) if list_b else set())

    scored: list[tuple[str, float, int]] = []
    for article in articles:
        rank_a = list_a.get(article, (None, None))[0]
        rank_b = list_b.get(article, (None, None))[0] if list_b else None

        score = 0.0
        if rank_a is not None:
            score += 1.0 / (rrf_k + rank_a)
        if rank_b is not None:
            score += 1.0 / (rrf_k + rank_b)

        min_rank = min(r for r in (rank_a, rank_b) if r is not None)
        scored.append((article, score, min_rank))

    scored.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [(article, score) for article, score, _ in scored]
