"""Tests for the yake_rust python package."""

from __future__ import annotations

import multiprocessing
import threading
import time
from typing import Any
from unittest.mock import Mock

import pytest
import yake as liaad_yake
from flaky import flaky

import yake_rust


@pytest.fixture
def mock_rust_function(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock_obj = Mock(spec=yake_rust._interface._get_n_best__rust)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "yake_rust._interface._get_n_best__rust",
        mock_obj,
    )
    return mock_obj


def test_instantiate_yake_with_language__unit() -> None:
    _ = yake_rust.Yake(yake_rust.YakeConfig(), language="en")


def test_instantiate_yake_with_custom_stopwords__unit() -> None:
    _ = yake_rust.Yake(yake_rust.YakeConfig(), stopwords={"stop", "word"})


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"language": "en", "stopwords": {"stopwords"}}, {"bad": 1}],
    ids=["no kwargs", "missing kwargs", "nonsense kwargs"],
)
def test_instantiate_yake_bad_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        _ = yake_rust.Yake(yake_rust.YakeConfig(), **kwargs)


def test_get_n_best_with_stopwords__unit(mock_rust_function: Mock) -> None:
    """Test get_n_best with the rust crate yake-rust mocked."""
    text = "this is a text"
    n = 2
    stopwords = {"stop"}
    config = Mock(spec=yake_rust.YakeConfig)

    actual = yake_rust.Yake(config, stopwords=stopwords).get_n_best(
        text,
        n=n,
    )
    mock_rust_function.assert_called_once_with(
        text,
        n,
        config.ngrams,
        config.punctuation,
        config.window_size,
        config.remove_duplicates,
        config.deduplication_threshold,
        config.strict_capital,
        config.only_alphanumeric_and_hyphen,
        config.minimum_chars,
        stopwords,
        None,
    )
    assert actual is mock_rust_function.return_value


def test_get_n_best_with_language__unit(mock_rust_function: Mock) -> None:
    """Test get_n_best with the rust crate yake-rust mocked."""
    text = "this is a text"
    n = 2
    language = "pt"
    config = Mock(spec=yake_rust.YakeConfig)

    actual = yake_rust.Yake(config, language=language).get_n_best(
        text,
        n=n,
    )
    mock_rust_function.assert_called_once_with(
        text,
        n,
        config.ngrams,
        config.punctuation,
        config.window_size,
        config.remove_duplicates,
        config.deduplication_threshold,
        config.strict_capital,
        config.only_alphanumeric_and_hyphen,
        config.minimum_chars,
        None,
        language,
    )
    assert actual is mock_rust_function.return_value


@pytest.mark.integration_test
@pytest.mark.parametrize(
    ("text", "n", "config", "expected"),
    [
        (
            "This is a keyword!",
            2,
            yake_rust.YakeConfig(),
            [("keyword", pytest.approx(0.1583, abs=0.001))],
        ),
        (
            "I will give you a great deal if you just read this.",
            2,
            yake_rust.YakeConfig(ngrams=2),
            [
                ("great deal", pytest.approx(0.0257, abs=0.001)),
                ("give", pytest.approx(0.1583, abs=0.001)),
            ],
        ),
        (
            "Greetings! Do you need new headphones? "
            "Good headphones? Like, really good headphones? "
            "I am a headphones salesperson!",
            1,
            yake_rust.YakeConfig(ngrams=1),
            [("headphones", pytest.approx(0.1694, abs=0.001))],
        ),
    ],
)
def test_get_n_best__integration(
    text: str,
    n: int,
    config: yake_rust.YakeConfig,
    expected: list[tuple[str, float]],
) -> None:
    """Test get_n_best with the underlying rust crate yake-rust."""
    actual = yake_rust.Yake(config, language="en").get_n_best(
        text,
        n=n,
    )
    assert actual == expected


@pytest.mark.integration_test
@pytest.mark.parametrize(
    ("text", "n", "ngrams"),
    [
        ("is your weekly newsletter!", 3, 2),
        ("This is a keyword!", 1, 1),
        (
            "This is your weekly newsletter! "
            "Hundreds of great deals - everything from men's fashion "
            "to high-tech drones!",
            2,
            1,
        ),
        ("I will give you a great deal if you just read this.", 10, 2),
        (
            "Greetings! Do you need new headphones? "
            "Good headphones? Like, really good headphones? "
            "I am a headphones salesperson!",
            1,
            1,
        ),
        (
            "I am a competition-centric person! "
            "I really like competition. "
            "Every competition is a hoot!",
            10,
            1,
        ),
    ],
)
def test_get_n_best__compare_with_liaad_yake(text: str, n: int, ngrams: int) -> None:
    """Test that results agree with the reference implementation."""
    liaad_extractor = liaad_yake.KeywordExtractor(
        lan="en",  # default, just making it explicit
        n=ngrams,
        dedupFunc="levenshtein",  # value used in yake-rust
        top=n,
    )
    liaad_result: list[tuple[str, float]] = liaad_extractor.extract_keywords(text)
    liaad_result = [
        (t[0], float(t[1])) for t in liaad_result
    ]  # LIAAD/yake returns np.float64

    config = yake_rust.YakeConfig(
        ngrams=ngrams,
    )
    our_yake = yake_rust.Yake(
        config,
        stopwords=liaad_extractor.stopword_set,  # use LIAAD/yake stopwords)
    )
    our_result: list[tuple[str, float]] = our_yake.get_n_best(
        text,
        n=n,
    )

    our_result_with_approximate_scores = [
        (word, pytest.approx(score, abs=0.001)) for (word, score) in our_result
    ]
    assert our_result_with_approximate_scores == liaad_result  # type: ignore[comparison-overlap]


LONG_TEXT = (
    "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, "
    "sed diam nonumy eirmod tempor invidunt ut "
    "labore et dolore magna aliquyam erat, sed diam voluptua. "
    "At vero eos et accusam et justo duo dolores et "
    "ea rebum. Stet clita kasd gubergren, no sea takimata "
    "sanctus est Lorem ipsum dolor sit amet. Lorem ipsum"
    " dolor sit amet, consetetur sadipscing elitr, sed diam "
    "nonumy eirmod tempor invidunt ut labore et dolore"
    " magna aliquyam erat, sed diam voluptua. At vero eos et "
    "accusam et justo duo dolores et ea rebum. "
    "Stet clita kasd gubergren, no sea takimata sanctus est "
    "Lorem ipsum dolor sit amet. "
) * 20


@pytest.mark.slow_integration_test
@flaky(  # type: ignore[misc]
    max_runs=10, min_passes=1
)  # If it passes even once, it must have been concurrent
def test_get_n_best__concurrency() -> None:
    """Test that keyword extraction can be concurrent (releases the GIL)."""
    if multiprocessing.cpu_count() < 2:
        pytest.skip("Only one cpu")

    # Long text to make it slow
    texts = [LONG_TEXT] * 4
    config = yake_rust.YakeConfig(window_size=2)
    yake = yake_rust.Yake(config, language="en")

    t0_sequential = time.perf_counter()
    for text in texts:
        _ = yake.get_n_best(text, n=1)
    dt_sequential = time.perf_counter() - t0_sequential

    threads: list[threading.Thread] = []
    for text in texts:
        threads.append(
            threading.Thread(
                target=yake.get_n_best,
                args=(text,),
                kwargs={"n": 1},
            )
        )
    t0_concurrent = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    dt_concurrent = time.perf_counter() - t0_concurrent

    # If the threads could run truly concurrently, it should have been significantly
    # faster than doing the same work concurrently (assuming multiple cpus):
    assert dt_concurrent <= dt_sequential * 0.5


def _run_yake_rust_and_liaad_yake(
    text: str, *, ngrams: int, n: int, window_size: int
) -> tuple[float, float]:
    """Returns a tuple (LIAAD/Yake result, yake-rust result)."""
    liaad_extractor = liaad_yake.KeywordExtractor(
        lan="en",  # default, just making it explicit
        n=ngrams,
        dedupFunc="levenshtein",  # value used in yake-rust
        windowsSize=window_size,
        top=n,
    )
    config = yake_rust.YakeConfig(
        ngrams=ngrams,
        remove_duplicates=True,  # always True for LIAAD/yake
        window_size=window_size,
    )
    yake_rust_extractor = yake_rust.Yake(
        config,
        stopwords=liaad_extractor.stopword_set,  # use LIAAD/yake stopwords
    )

    t0_liaad = time.perf_counter()
    _ = liaad_extractor.extract_keywords(text)
    dt_liaad = time.perf_counter() - t0_liaad

    t0_rust = time.perf_counter()
    _ = yake_rust_extractor.get_n_best(text, n=n)
    dt_rust = time.perf_counter() - t0_rust

    return dt_liaad, dt_rust


@pytest.mark.integration_test
@flaky(max_runs=10, min_passes=6)  # type: ignore[misc]
def test_get_n_best__race_liaad_yake__short_text() -> None:
    dt_liaad, dt_rust = _run_yake_rust_and_liaad_yake(
        "short text",
        ngrams=1,
        n=1,
        window_size=1,
    )
    # yake-rust is not faster for a short text, but it
    # is not allowed to be that much slower either
    assert dt_rust - dt_liaad < 0.01


@pytest.mark.slow_integration_test
@flaky(max_runs=10, min_passes=6)  # type: ignore[misc]
def test_get_n_best__race_liaad_yake__long_text() -> None:
    dt_liaad, dt_rust = _run_yake_rust_and_liaad_yake(
        LONG_TEXT,
        ngrams=3,  # Make it more demanding
        n=10,
        window_size=4,  # Make it more demanding
    )
    assert dt_rust < dt_liaad
