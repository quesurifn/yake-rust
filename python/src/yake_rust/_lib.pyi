"""Python stubs for _lib.rs."""

from __future__ import annotations

from typing import overload

class Yake:
    @overload
    def __init__(
        self,
        *,
        ngrams: int | None,
        punctuation: set[str] | None,
        window_size: int | None,
        remove_duplicates: bool | None,
        deduplication_threshold: float | None,
        strict_capital: bool | None,
        only_alphanumeric_and_hyphen: bool | None,
        minimum_chars: int | None,
        stopwords: set[str],
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        ngrams: int | None,
        punctuation: set[str] | None,
        window_size: int | None,
        remove_duplicates: bool | None,
        deduplication_threshold: float | None,
        strict_capital: bool | None,
        only_alphanumeric_and_hyphen: bool | None,
        minimum_chars: int | None,
        language: str,
    ) -> None: ...
    def __init__(
        self,
        *,
        ngrams: int | None = None,
        punctuation: set[str] | None = None,
        window_size: int | None = None,
        remove_duplicates: bool | None = None,
        deduplication_threshold: float | None = None,
        strict_capital: bool | None = None,
        only_alphanumeric_and_hyphen: bool | None = None,
        minimum_chars: int | None = None,
        stopwords: set[str] | None = None,
        language: str | None = None,
    ) -> None: ...
    def get_n_best(self, text: str, /, *, n: int) -> list[tuple(str, float)]: ...
