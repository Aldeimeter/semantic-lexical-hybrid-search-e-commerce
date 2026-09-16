import argparse
import csv
import os
from dataclasses import dataclass

from search.catalog import load_catalog
from search.fusion import rrf_merge
from search.pipeline import SearchPipeline, build_pipeline

RRF_K_SWEEP = [0, 1, 2, 3, 4, 5]  # k>5 не проверяем — монотонно хуже Hit@3 без выигрыша в Precision@1, см. ARCHITECTURE.md


@dataclass
class QueryCase:
    id: int
    query: str
    expected_article: str


@dataclass
class EvalRow:
    """Одна строка на (запрос, подход) — для агрегатов и для ручного
    разбора в Google Sheets. Никакой категоризации причин промахов
    или разметки типа запроса здесь не считается — это ручная работа."""

    query_id: int
    query: str
    expected_article: str
    approach: str  # "fuzzy_lexical" | "vector_only" | "hybrid"
    rank_in_lexical: int | None
    rank_in_vector: int | None
    rank_in_final: int | None
    hit_at_1: bool
    hit_at_3: bool


def load_queries(csv_path: str) -> list[QueryCase]:
    cases = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            cases.append(
                QueryCase(
                    id=int(row["id"]),
                    query=row["Запрос"],
                    expected_article=row["Ожидаемый артикул"],
                )
            )
    return cases


def evaluate(pipeline: SearchPipeline, queries: list[QueryCase], approach: str) -> list[EvalRow]:
    """approach:
    - "fuzzy_lexical" -> только list_a (без вектора)
    - "vector_only"   -> только list_b (без лексики/fuzzy вообще, диагностика)
    - "hybrid"         -> RRF-слияние list_a и list_b

    Для каждого query: pipeline.search_full(query.query, use_semantic=...) ->
    SearchDebugInfo(list_a, list_b, merged).
    rank_in_final=None означает "не найден ни одной веткой вообще", не
    "за пределами top_k" — top_k здесь не участвует, merged несрезан."""
    use_semantic = approach in ("vector_only", "hybrid")
    rows = []
    for case in queries:
        debug = pipeline.search_full(case.query, use_semantic=use_semantic)

        rank_lex = debug.list_a.get(case.expected_article, (None, None))[0]
        rank_vec = debug.list_b.get(case.expected_article, (None, None))[0] if debug.list_b else None

        if approach == "vector_only":
            rank_final = rank_vec
        else:
            merged_articles = [article for article, _ in debug.merged]
            rank_final = (
                merged_articles.index(case.expected_article) + 1
                if case.expected_article in merged_articles
                else None
            )

        rows.append(
            EvalRow(
                query_id=case.id,
                query=case.query,
                expected_article=case.expected_article,
                approach=approach,
                rank_in_lexical=rank_lex,
                rank_in_vector=rank_vec,
                rank_in_final=rank_final,
                hit_at_1=(rank_final == 1),
                hit_at_3=(rank_final is not None and rank_final <= 3),
            )
        )
    return rows


def summarize(rows: list[EvalRow]) -> dict[str, dict[str, float]]:
    """approach -> {"precision_at_1": ..., "hit_at_3": ...}"""
    by_approach: dict[str, list[EvalRow]] = {}
    for row in rows:
        by_approach.setdefault(row.approach, []).append(row)

    result = {}
    for approach, approach_rows in by_approach.items():
        n = len(approach_rows)
        result[approach] = {
            "precision_at_1": sum(r.hit_at_1 for r in approach_rows) / n,
            "hit_at_3": sum(r.hit_at_3 for r in approach_rows) / n,
        }
    return result


def sweep_rrf_k(pipeline: SearchPipeline, queries: list[QueryCase], rrf_k_values: list[float]) -> dict[float, dict[str, float]]:
    """Свип rrf_k без повторных сетевых вызовов к TEI: list_a/list_b считаются
    ОДИН раз на запрос (use_semantic=True даёт оба), дальше для каждого k из
    свипа просто пересчитывается rrf_merge (чистая функция, без сети)."""
    cached = []
    for case in queries:
        debug = pipeline.search_full(case.query, use_semantic=True)
        cached.append((case, debug.list_a, debug.list_b))

    results: dict[float, dict[str, float]] = {}
    for k in rrf_k_values:
        hits1 = hits3 = 0
        for case, list_a, list_b in cached:
            merged = rrf_merge(list_a, list_b, k)
            articles = [article for article, _ in merged]
            rank = articles.index(case.expected_article) + 1 if case.expected_article in articles else None
            hits1 += rank == 1
            hits3 += rank is not None and rank <= 3
        n = len(cached)
        results[k] = {"precision_at_1": hits1 / n, "hit_at_3": hits3 / n}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rrf-k", type=float, default=3)
    parser.add_argument("--sweep", action="store_true", help="прогнать RRF_K_SWEEP вместо одного --rrf-k")
    parser.add_argument("--enrichment", type=str, default=None, help="путь к enrichment.csv (Артикул;Сценарий), опционально")
    args = parser.parse_args()

    products = load_catalog("data/products.csv")
    queries = load_queries("data/queries.csv")

    embeddings_url = os.environ.get("EMBEDDINGS_URL", "http://localhost:8080")
    pipeline = build_pipeline(products, embeddings_url, rrf_k=args.rrf_k, enrichment_path=args.enrichment)

    if args.sweep:
        sweep = sweep_rrf_k(pipeline, queries, RRF_K_SWEEP)
        print(f"{'rrf_k':>6s} {'Hybrid Precision@1':>20s} {'Hybrid Hit@3':>13s}")
        for k, metrics in sweep.items():
            print(f"{k:6g} {metrics['precision_at_1'] * 100:19.1f}% {metrics['hit_at_3'] * 100:12.1f}%")
        return

    rows: list[EvalRow] = []
    rows += evaluate(pipeline, queries, "fuzzy_lexical")
    rows += evaluate(pipeline, queries, "vector_only")
    rows += evaluate(pipeline, queries, "hybrid")

    summary = summarize(rows)

    print(f"{'подход':16s} {'Precision@1':>12s} {'Hit@3':>8s}")
    for approach in ("fuzzy_lexical", "vector_only", "hybrid"):
        metrics = summary[approach]
        print(f"{approach:16s} {metrics['precision_at_1'] * 100:11.1f}% {metrics['hit_at_3'] * 100:7.1f}%")

    out_path = "eval_results.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "query_id", "query", "expected_article", "approach",
                "rank_in_lexical", "rank_in_vector", "rank_in_final",
                "hit_at_1", "hit_at_3",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.query_id, r.query, r.expected_article, r.approach,
                    r.rank_in_lexical, r.rank_in_vector, r.rank_in_final,
                    r.hit_at_1, r.hit_at_3,
                ]
            )
    print(f"\nПо-запросные данные сохранены в {out_path} — для ручного разбора в Google Sheets.")


if __name__ == "__main__":
    main()
