import re
from typing import TYPE_CHECKING

import pymorphy3

if TYPE_CHECKING:
    from .catalog import Product

_morph = pymorphy3.MorphAnalyzer()

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Нижний регистр, убрать пунктуацию, разбить на слова."""
    return _TOKEN_RE.findall(text.lower())


def lemmatize(tokens: list[str]) -> list[str]:
    """pymorphy3, лучший разбор на каждый токен.

    Порядок в пайплайне: токенизация -> лемматизация -> OOV-проверка -> fuzzy
    (см. ARCHITECTURE.md п.4 — НЕ fuzzy до лемматизации: pymorphy3 не
    'ломается' на опечатках, предсказывает лемму по суффиксам даже для
    неизвестных слов)."""
    return [_morph.parse(token)[0].normal_form for token in tokens]


def lemmatize_catalog(products: list["Product"]) -> list[list[str]]:
    """Токенизирует+лемматизирует Наименование каждого товара — ОДИН раз,
    единственный проход по каталогу. Результат (параллельный products по
    индексу) передаётся и во FuzzyDictionary.build(), и в
    LexicalSearcher.build() — чтобы оба строились из идентичных лемм."""
    return [lemmatize(tokenize(product.name)) for product in products]
