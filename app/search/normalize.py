import re
from typing import TYPE_CHECKING

import pymorphy3

if TYPE_CHECKING:
    from .catalog import Product

_morph = pymorphy3.MorphAnalyzer()

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

# Точечные сокращения каталога, которые pymorphy3 не может разобрать
# корректно — токенизатор режет их по "." на обрубки, и лемматизатор
# угадывает им лемму из не относящегося к делу слова (например "лам" из
# "мат.лам."/"глянц.лам." -> "лама", животное). Раскрываем ДО токенизации,
# на уровне текста, а не выкидываем — это содержательные слова
# ("матовый"/"глянцевый"/"подарочный"/"тиснение"), которые реально могут
# встретиться в запросе, в отличие от единиц измерения ниже. Найдено сканом
# каталога (см. ISSUES.md п.1) — список расширяется по мере новых находок,
# не пытаемся угадать заранее все возможные сокращения.
_ABBREVIATIONS: dict[str, str] = {
    "мат.лам.": "матовая ламинация",
    "глянц.лам.": "глянцевая ламинация",
    "подар.": "подарочный",
    "тисн.": "тиснение",
}

# Единицы измерения — исключаются как токены целиком после разбивки: не
# несут поискового смысла, а некоторые ложно лемматизируются в случайные
# слова (например "см" -> "смотреть", глагол). См. ISSUES.md п.2.
_UNIT_STOPWORDS: set[str] = {"см", "мм", "шт", "г", "кг", "л", "мл"}


def tokenize(text: str) -> list[str]:
    """Нижний регистр, раскрытие известных сокращений, убрать пунктуацию,
    разбить на слова, отфильтровать единицы измерения."""
    text = text.lower()
    for abbreviation, expansion in _ABBREVIATIONS.items():
        text = text.replace(abbreviation, expansion)
    tokens = _TOKEN_RE.findall(text)
    return [token for token in tokens if token not in _UNIT_STOPWORDS]


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
