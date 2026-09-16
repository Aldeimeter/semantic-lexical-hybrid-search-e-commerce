import csv
from dataclasses import dataclass


@dataclass
class Product:
    article: str
    category: str
    name: str
    price: float
    stock: int
    # Текст для индексации (лексика/вектор) — по умолчанию = name. Если
    # применено обогащение (см. apply_enrichment), содержит name + сценарий.
    # name остаётся чистым Наименованием для отображения в SearchResult —
    # обогащение никогда не подмешивается туда, где его видит пользователь.
    search_text: str = ""

    def __post_init__(self):
        if not self.search_text:
            self.search_text = self.name


def load_catalog(csv_path: str) -> list[Product]:
    """Читает products.csv (разделитель ';'), возвращает список Product.
    products.csv выгружается из 1С заново каждый день — этот файл никогда
    не редактируется нами напрямую, поэтому обогащение (apply_enrichment)
    живёт в отдельном файле и джойнится по Артикулу после загрузки, а не
    записывается в сам products.csv."""
    products = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            products.append(
                Product(
                    article=row["Артикул"],
                    category=row["Категория"],
                    name=row["Наименование"],
                    price=float(row["Цена"]),
                    stock=int(row["Остаток"]),
                )
            )
    return products


def load_enrichment(csv_path: str) -> dict[str, str]:
    """Читает enrichment.csv (Артикул;Сценарий) -> {article: scenario_text}."""
    result = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            result[row["Артикул"]] = row["Сценарий"]
    return result


def apply_enrichment(products: list[Product], enrichment: dict[str, str]) -> None:
    """Джойн по Артикулу: если для товара есть сценарий, добавляет его к
    search_text (name остаётся нетронутым). Товары без записи в enrichment
    (новые в 1С, ещё не обогащённые) — просто не меняются, деградация без
    ошибок."""
    for product in products:
        scenario = enrichment.get(product.article)
        if scenario:
            product.search_text = f"{product.name} {scenario}"
