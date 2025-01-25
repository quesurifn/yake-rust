use std::collections::HashSet;

use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (text, n, ngrams, punctuation, window_size, remove_duplicates, deduplication_threshold, strict_capital, only_alphanumeric_and_hyphen, minimum_chars, stopwords, language, /))]
fn get_n_best(
    py: Python<'_>,
    text: String,
    n: usize,
    ngrams: Option<usize>,
    punctuation: Option<HashSet<char>>,
    window_size: Option<usize>,
    remove_duplicates: Option<bool>,
    deduplication_threshold: Option<f64>,
    strict_capital: Option<bool>,
    only_alphanumeric_and_hyphen: Option<bool>,
    minimum_chars: Option<usize>,
    stopwords: Option<HashSet<String>>,
    language: Option<String>,
) -> Vec<(String, f64)> {
    py.allow_threads(|| {
        get_n_best_sequential(
            text,
            n,
            ngrams,
            punctuation,
            window_size,
            remove_duplicates,
            deduplication_threshold,
            strict_capital,
            only_alphanumeric_and_hyphen,
            minimum_chars,
            stopwords,
            language,
        )
    })
}

fn get_n_best_sequential(
    text: String,
    n: usize,
    ngrams: Option<usize>,
    punctuation: Option<HashSet<char>>,
    window_size: Option<usize>,
    remove_duplicates: Option<bool>,
    deduplication_threshold: Option<f64>,
    strict_capital: Option<bool>,
    only_alphanumeric_and_hyphen: Option<bool>,
    minimum_chars: Option<usize>,
    stopwords: Option<HashSet<String>>,
    language: Option<String>,
) -> Vec<(String, f64)> {
    let default_config = yake_rust::Config::default();
    let mut used_stopwords: Option<yake_rust::StopWords> = None;
    // TODO: Can we avoid reloading stopwords every time?
    if stopwords.is_some() {
        used_stopwords = Some(yake_rust::StopWords::custom(stopwords.unwrap()));
    } else {
        used_stopwords = yake_rust::StopWords::predefined(&language.unwrap());
    }
    let config = yake_rust::Config {
        ngrams: ngrams.unwrap_or(default_config.ngrams),
        punctuation: punctuation.unwrap_or(default_config.punctuation),
        window_size: window_size.unwrap_or(default_config.window_size),
        remove_duplicates: remove_duplicates.unwrap_or(default_config.remove_duplicates),
        deduplication_threshold: deduplication_threshold.unwrap_or(default_config.deduplication_threshold),
        strict_capital: strict_capital.unwrap_or(default_config.strict_capital),
        only_alphanumeric_and_hyphen: only_alphanumeric_and_hyphen
            .unwrap_or(default_config.only_alphanumeric_and_hyphen),
        minimum_chars: minimum_chars.unwrap_or(default_config.minimum_chars),
    };
    let results = yake_rust::get_n_best(n, &text, &used_stopwords.expect("no stopwords"), &config);
    results.iter().map(|r: &yake_rust::ResultItem| (r.raw.to_string(), r.score)).collect::<Vec<_>>()
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_n_best, m)?)
}
