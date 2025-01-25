"""Python stubs for _lib.rs."""

from __future__ import annotations

class Yake:
    def __init__(
        self,
        *,
        stopwords: set[str] | None = None,
        language: str | None = None,
        ngrams: int | None = None,
        punctuation: set[str] | None = None,
        window_size: int | None = None,
        remove_duplicates: bool | None = None,
        deduplication_threshold: float | None = None,
        strict_capital: bool | None = None,
        only_alphanumeric_and_hyphen: bool | None = None,
        minimum_chars: int | None = None,
    ) -> None: ...
    def get_n_best(self, text: str, *, n: int) -> list[tuple[str, float]]: ...
