from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


class FuzzyDictionary:
    """Словарь уникальных ЛЕММ каталога (из поля Наименование). Без частоты —
    правило коррекции ("ровно один кандидат прошёл порог") её не использует."""

    def __init__(self, lemmas: set[str]):
        self._lemmas = lemmas
        self._lemmas_list = list(lemmas)  # кэш для итерации в FuzzyCorrector

    @classmethod
    def build(cls, product_lemmas: list[list[str]]) -> "FuzzyDictionary":
        """product_lemmas — результат lemmatize_catalog(products), тот же
        самый вызов, что передаётся в LexicalSearcher.build()."""
        lemmas: set[str] = set()
        for lemma_list in product_lemmas:
            lemmas.update(lemma_list)
        return cls(lemmas)

    def __contains__(self, lemma: str) -> bool:
        return lemma in self._lemmas

    def all_lemmas(self) -> list[str]:
        return self._lemmas_list


def _max_edit_distance(word: str) -> int:
    """Токены <=6 букв: максимум 1 правка; >6 букв: максимум 2."""
    return 1 if len(word) <= 6 else 2


class FuzzyCorrector:
    """threshold=80.0 (rapidfuzz.fuzz.ratio).

    correct() — единственное место, где принимается решение об OOV и
    коррекции; вызывающий код (SearchPipeline) НЕ проверяет OOV сам, просто
    вызывает correct() для каждой леммы без условий."""

    def __init__(self, dictionary: FuzzyDictionary, threshold: float = 80.0):
        self._dictionary = dictionary
        self._threshold = threshold

    def correct(self, lemma: str) -> str:
        """Возвращает исправленную лемму или исходную без изменений.
        Безопасно вызывать для ЛЮБОЙ леммы, включая уже валидные — OOV-проверка
        находится внутри, не снаружи."""
        if lemma in self._dictionary:
            return lemma

        candidates = [
            vocab_word
            for vocab_word in self._dictionary.all_lemmas()
            if fuzz.ratio(lemma, vocab_word) >= self._threshold
        ]
        if len(candidates) != 1:
            # ноль или несколько кандидатов — неоднозначность, не исправляем
            return lemma

        candidate = candidates[0]
        if Levenshtein.distance(lemma, candidate) > _max_edit_distance(lemma):
            return lemma
        return candidate
