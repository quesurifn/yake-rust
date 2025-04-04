use std::collections::HashSet;
use std::ops::Deref;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[pyclass(module = "yake_rust", frozen)]
struct Yake {
    _config: yake_rust::Config,
    _stopwords: yake_rust::StopWords,
}

#[pymethods]
impl Yake {
    #[new]
    #[pyo3(signature = (stopwords=None, language=None, ngrams=None, punctuation=None, window_size=None, remove_duplicates=None, deduplication_threshold=None, strict_capital=None, only_alphanumeric_and_hyphen=None, minimum_chars=None))]
    fn new(
        stopwords: Option<HashSet<String>>,
        language: Option<String>,
        ngrams: Option<usize>,
        punctuation: Option<HashSet<char>>,
        window_size: Option<usize>,
        remove_duplicates: Option<bool>,
        deduplication_threshold: Option<f64>,
        strict_capital: Option<bool>,
        only_alphanumeric_and_hyphen: Option<bool>,
        minimum_chars: Option<usize>,
    ) -> PyResult<Self> {
        let default_config = yake_rust::Config::default();
        if stopwords.is_none() == language.is_none() {
            return Err(PyTypeError::new_err("Provide either language or stopwords, but not both."));
        }
        let used_stopwords: Option<yake_rust::StopWords> = {
            if stopwords.is_some() {
                Some(yake_rust::StopWords::custom(stopwords.unwrap()))
            } else {
                yake_rust::StopWords::predefined(&language.unwrap())
            }
        };
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
        Ok(Yake { _config: config, _stopwords: used_stopwords.unwrap() })
    }

    /// get_n_best(self, text, *, n)
    /// --
    ///
    /// Get the n best keywords from text.
    #[pyo3(signature = (text, *, n))]
    pub fn get_n_best(&self, py: Python, text: &str, n: usize) -> Vec<(String, f64)> {
        py.allow_threads(|| get_n_best_sequential(n, text, &self._stopwords, &self._config))
    }

    #[getter]
    fn get_stopwords(&self) -> PyResult<HashSet<String>> {
        Ok(HashSet::from_iter(self._stopwords.iter().cloned()))
    }
}

fn get_n_best_sequential(
    n: usize,
    text: &str,
    stopwords: &yake_rust::StopWords,
    config: &yake_rust::Config,
) -> Vec<(String, f64)> {
    let results = yake_rust::get_n_best(n, &text, stopwords, config);
    results.iter().map(|r: &yake_rust::ResultItem| (r.raw.to_string(), r.score)).collect::<Vec<_>>()
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Yake>()?;
    Ok(())
}
