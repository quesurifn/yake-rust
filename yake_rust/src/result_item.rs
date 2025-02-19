use std::cmp::max;

use levenshtein::levenshtein;

use crate::{Candidate, LTerm};

/// Represents a key phrase.
#[derive(PartialEq, Clone, Debug)]
pub struct ResultItem {
    /// The first occurrence in the text. Not exact, as words are joined by a single space.
    pub raw: String,
    /// A lowercased key phrase consisting of 1…N words, where N is configured through [`Config::ngrams`].
    pub keyword: LTerm,
    /// Key importance, where 0 is the most important.
    pub score: f64,
}

impl PartialEq<(&str, &str, f64)> for ResultItem {
    fn eq(&self, (raw, key_phrase, score): &(&str, &str, f64)) -> bool {
        self.raw.eq(raw) && self.keyword.eq(key_phrase) && self.score.eq(score)
    }
}

impl From<Candidate<'_>> for ResultItem {
    fn from(candidate: Candidate) -> Self {
        ResultItem { raw: candidate.raw.join(" "), keyword: candidate.lc_terms.join(" "), score: candidate.score }
    }
}

pub(crate) fn remove_duplicates(threshold: f64, results: Vec<ResultItem>, n: usize) -> Vec<ResultItem> {
    let mut unique: Vec<ResultItem> = Vec::new();

    for res in results {
        if unique.len() >= n {
            break;
        }

        let is_duplicate = unique.iter().any(|it| levenshtein_ratio(&it.keyword, &res.keyword) > threshold);

        if !is_duplicate {
            unique.push(res);
        }
    }

    unique
}

/// Returns a number in 0..1 range, where 0 is distant and 1 is close.
fn levenshtein_ratio(seq1: &str, seq2: &str) -> f64 {
    let distance = if seq1.len() <= seq2.len() { levenshtein(seq1, seq2) } else { levenshtein(seq2, seq1) };
    let length = max(seq1.len(), seq2.len());
    1.0 - (distance as f64 / length as f64)
}
