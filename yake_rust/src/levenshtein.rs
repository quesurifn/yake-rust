use std::cmp::max;

use natural::distance::levenshtein_distance;

pub(crate) struct Levenshtein {}
impl Levenshtein {
    pub fn ratio(seq1: String, seq2: String) -> f64 {
        let mut distance = 0;
        if seq1.len() <= seq2.len() {
            distance = levenshtein_distance(&seq1, &seq2);
        } else {
            distance = levenshtein_distance(&seq2, &seq1);
        }
        let length = max(seq1.len(), seq2.len());
        1.0 - (distance as f64 / length as f64)
    }
}
