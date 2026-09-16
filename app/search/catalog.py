import csv
from dataclasses import dataclass


@dataclass
class Product:
    article: str
    category: str
    name: str
    price: float
    stock: int


def load_catalog(csv_path: str) -> list[Product]:
    """Читает products.csv (разделитель ';'), возвращает список Product."""
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
