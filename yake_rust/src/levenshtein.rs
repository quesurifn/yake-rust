use std::cmp::max;

use levenshtein::levenshtein;

/// Returns a number in 0..1 range, where 0 is distant and 1 is close.
pub fn levenshtein_ratio(seq1: &str, seq2: &str) -> f64 {
    let distance = if seq1.len() <= seq2.len() { levenshtein(seq1, seq2) } else { levenshtein(seq2, seq1) };
    let length = max(seq1.len(), seq2.len());
    1.0 - (distance as f64 / length as f64)
}
