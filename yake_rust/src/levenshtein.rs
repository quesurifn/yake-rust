use std::cmp::max;

use levenshtein::levenshtein;

pub(crate) struct Levenshtein {}

impl Levenshtein {
    pub fn ratio(seq1: String, seq2: String) -> f64 {
        let distance = if seq1.len() <= seq2.len() { levenshtein(&seq1, &seq2) } else { levenshtein(&seq2, &seq1) };
        let length = max(seq1.len(), seq2.len());
        1.0 - (distance as f64 / length as f64)
    }
}
