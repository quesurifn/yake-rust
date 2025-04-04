"""Tests for the yake_rust python package."""

from __future__ import annotations

import multiprocessing
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yake as liaad_yake
from flaky import flaky

import yake_rust


@pytest.fixture(scope="session", autouse=True)
def _warm_up() -> None:
    """Run keyword extraction once to initialize segtok."""
    # First run of yake-rust is slow because of segtok.
    _ = yake_rust.Yake(language="en").get_n_best("Hello world!", n=1)


def test_instantiate_yake_with_language() -> None:
    _ = yake_rust.Yake(language="en")


def test_instantiate_yake_with_custom_stopwords() -> None:
    _ = yake_rust.Yake(stopwords={"stop", "word"})


def test_yake_stopwords_attribute__custom() -> None:
    stopwords = {"stop", "word"}
    yake = yake_rust.Yake(stopwords=stopwords)
    assert yake.stopwords == stopwords


def test_yake_stopwords_attribute__language() -> None:
    yake = yake_rust.Yake(language="en")
    assert yake.stopwords


def test_yake_stopwords_attribute__read_only() -> None:
    yake = yake_rust.Yake(language="en")
    with pytest.raises(AttributeError, match="stopwords"):
        yake.stopwords = set()  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"language": "en", "stopwords": {"stopwords"}}, {"bad": 1}],
    ids=["no kwargs", "missing kwargs", "nonsense kwargs"],
)
def test_instantiate_yake_bad_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        _ = yake_rust.Yake(**kwargs)


@pytest.mark.parametrize(
    ("text", "n", "kwargs", "expected"),
    [
        (
            "This is a keyword!",
            2,
            {"language": "en"},
            [("keyword", pytest.approx(0.1583, abs=0.001))],
        ),
        (
            "I will give you a great deal if you just read this.",
            2,
            {"ngrams": 2, "language": "en"},
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
            {"ngrams": 1, "language": "en"},
            [("headphones", pytest.approx(0.1694, abs=0.001))],
        ),
    ],
)
def test_get_n_best(
    text: str,
    n: int,
    kwargs: dict[str, Any],
    expected: list[tuple[str, float]],
) -> None:
    """Test get_n_best with the underlying rust crate yake-rust."""
    actual = yake_rust.Yake(**kwargs).get_n_best(
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

    our_yake = yake_rust.Yake(
        ngrams=ngrams,
        stopwords=liaad_extractor.stopword_set,  # use LIAAD/yake stopwords
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


@pytest.mark.integration_test
@flaky(max_runs=10, min_passes=2)  # type: ignore[misc]
def test_get_n_best__concurrency() -> None:
    """Test that keyword extraction can be concurrent (releases the GIL)."""
    if multiprocessing.cpu_count() < 2:
        pytest.skip("Only one cpu")

    # Long text to make it slow
    texts = [LONG_TEXT] * 4
    results: list[list[tuple[str, float]]] = [[]] * len(texts)
    expected_result = [("Lorem ipsum dolor", pytest.approx(0.000436, abs=0.0001))]
    yake = yake_rust.Yake(ngrams=3, window_size=2, language="en")

    t0_sequential = time.perf_counter()
    for index, text in enumerate(texts):
        results[index] = yake.get_n_best(text, n=1)
    dt_sequential = time.perf_counter() - t0_sequential

    for result in results:
        assert result == expected_result  # type: ignore[comparison-overlap]

    times: list[tuple[float, float]] = [(-1.0, -1.0)] * len(texts)

    def place_result_in_list(i: int, s: str) -> None:
        start_t = time.monotonic_ns()
        results[i] = yake.get_n_best(s, n=1)
        end_t = time.monotonic_ns()
        times[i] = (start_t, end_t)

    results = [[]] * len(texts)
    threads: list[threading.Thread] = []
    for index, text in enumerate(texts):
        threads.append(
            threading.Thread(
                target=place_result_in_list,
                args=(index, text),
            )
        )
    t0_concurrent = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    dt_concurrent = time.perf_counter() - t0_concurrent

    for result in results:
        assert result == expected_result  # type: ignore[comparison-overlap]

    start_times = [t[0] for t in times]
    end_times = [t[1] for t in times]
    # All threads started before any thread ended:
    assert all(
        all(start_time < end_time for end_time in end_times)
        for start_time in start_times
    )

    # If the threads could run truly concurrently, it should have been significantly
    # faster than doing the same work concurrently (assuming multiple cpus).
    # However, this is a bit flaky, so give it a bit of a leeway.
    assert dt_concurrent <= dt_sequential * 0.65


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
    yake_rust_extractor = yake_rust.Yake(
        ngrams=ngrams,
        remove_duplicates=True,  # always True for LIAAD/yake
        window_size=window_size,
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
def test_get_n_best__race_liaad_yake__short_text() -> None:
    dt_liaad, dt_rust = _run_yake_rust_and_liaad_yake(
        "short text",
        ngrams=1,
        n=1,
        window_size=1,
    )
    assert dt_rust < dt_liaad


@pytest.mark.integration_test
def test_get_n_best__race_liaad_yake__long_text() -> None:
    dt_liaad, dt_rust = _run_yake_rust_and_liaad_yake(
        LONG_TEXT,
        ngrams=3,  # Make it more demanding
        n=10,
        window_size=4,  # Make it more demanding
    )
    assert dt_rust < dt_liaad


@pytest.mark.benchmark
def test_compare_benchmark_with_liaad_yake(capsys: pytest.CaptureFixture[str]) -> None:
    N = 100
    with open(
        Path(__file__).parent.parent.parent / "yake_rust" / "benches" / "100kb.txt"
    ) as filehandle:
        text = filehandle.read()
    ngrams = 3
    nr_of_words = 10
    window_size = 4

    liaad_extractor = liaad_yake.KeywordExtractor(
        lan="en",  # default, just making it explicit
        n=ngrams,
        dedupFunc="levenshtein",  # value used in yake-rust
        windowsSize=window_size,
        top=nr_of_words,
    )
    yake_rust_extractor = yake_rust.Yake(
        ngrams=ngrams,
        remove_duplicates=True,  # always True for LIAAD/yake
        window_size=window_size,
        stopwords=liaad_extractor.stopword_set,  # use LIAAD/yake stopwords
    )

    dts_liaad: list[float] = []
    for _ in range(N):
        t0_liaad = time.perf_counter()
        _ = liaad_extractor.extract_keywords(text)
        dt_liaad = time.perf_counter() - t0_liaad
        dts_liaad.append(dt_liaad)
    mean_liaad = statistics.mean(dts_liaad)
    median_liaad = statistics.median(dts_liaad)
    std_liaad = statistics.stdev(dts_liaad)

    dts_rust: list[float] = []
    for _ in range(N):
        t0_rust = time.perf_counter()
        _ = yake_rust_extractor.get_n_best(text, n=nr_of_words)
        dt_rust = time.perf_counter() - t0_rust
        dts_rust.append(dt_rust)
    mean_rust = statistics.mean(dts_rust)
    median_rust = statistics.median(dts_rust)
    std_rust = statistics.stdev(dts_rust)

    with capsys.disabled():
        print("")
        print(f"{mean_liaad=:.5}, {median_liaad=:.5}, {std_liaad=:.5}")
        print(f"{mean_rust=:.5}, {median_rust=:.5}, {std_rust=:.5}")
        print(
            f"yake-rust took {mean_rust / mean_liaad:.0%} of the time LIAAD/yake took"
        )
        print("")
