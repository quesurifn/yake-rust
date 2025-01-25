"""Python interface to yake-rust."""

from __future__ import annotations

import dataclasses
import sys
from typing import Any, overload

from ._lib import get_n_best as _get_n_best__rust

dataclass_kwargs: dict[str, Any] = {}
if sys.version_info >= (3, 10):
    # slots option available starting Python 3.10 (faster access, less memory)
    dataclass_kwargs["slots"] = True


@dataclasses.dataclass(**dataclass_kwargs)
class YakeConfig:
    """
    Configuration for YAKE.

    Leaving fields as None will result in the yake-rust default
    being used.

    Attributes:
        ngrams: The max number of ngrams in a keyword.
        punctuation: Set of punctuation symbols.
        window_size: The nr of tokens that are considered as a context.
        remove_duplicates: Whether to remove duplicate keywords.
        deduplication_threshold: The numerical threshold (Levenshtein distance)
            for considering words to be duplicates.
        strict_capital: Calculate the "term casing" metric by counting
            capitalized terms without intermediate uppercase letters.
        only_alphanumeric_and_hyphen: Key phrases are allowed to only have
            alphanumeric characters and hyphens.
        minimum_chars: Minimum length of key phrases.
        language: The language to use - selects stopwords from the default list.

    """

    ngrams: int | None = None
    punctuation: set[str] | None = None
    window_size: int | None = None
    remove_duplicates: bool | None = None
    deduplication_threshold: float | None = None
    strict_capital: bool | None = None
    only_alphanumeric_and_hyphen: bool | None = None
    minimum_chars: int | None = None


class Yake:
    """Keyword extractor.

    Example usage:

    >>> yake = Yake(YakeConfig(), language="en")
    >>> yake.get_n_best("One smartwatch. One phone. Many phones.", n=1)
    [("smartwatch", 0.2025)]

    """

    __slots__ = ("_config", "_language", "_stopwords")

    @overload
    def __init__(self, config: YakeConfig, *, stopwords: set[str]) -> None: ...

    @overload
    def __init__(self, config: YakeConfig, *, language: str) -> None: ...

    def __init__(
        self,
        config: YakeConfig,
        *,
        stopwords: set[str] | None = None,
        language: str | None = None,
    ) -> None:
        """Initialize an instance.

        Args:
            config: Configuration for YAKE.
            stopwords: Custom stopwords to use. Incompatible with "language".
                Either "stopwords" or "language" must be given.
            language: ISO 639 abbreviation for a language, e.g., "en".
                Sets the stopwords to the default stopwords for that language.
                Incompatible with "language". Either "stopwords" or "language"
                must be given.

        """
        if (stopwords is not None) == (language is not None):
            raise TypeError("Exactly one of 'stopwords' and 'language' must be given.")
        self._config = config or YakeConfig()
        self._stopwords = stopwords
        self._language = language

    def get_n_best(
        self,
        text: str,
        /,
        *,
        n: int,
    ) -> list[tuple[str, float]]:
        """
        Get the n best keywords from a text.

        Args:
            text: The text to extract keywords from.
            n: The number of keywords to return (top n lowest score).

        """
        return _get_n_best__rust(
            text,
            n,
            self._config.ngrams,
            self._config.punctuation,
            self._config.window_size,
            self._config.remove_duplicates,
            self._config.deduplication_threshold,
            self._config.strict_capital,
            self._config.only_alphanumeric_and_hyphen,
            self._config.minimum_chars,
            self._stopwords,
            self._language,
        )
