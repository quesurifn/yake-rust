use hashbrown::HashMap;

use crate::counter::Counter;
use crate::UTerm;

/// Stats for a single term `T` against another terms.
#[derive(Default)]
pub struct PairwiseFreq<'s> {
    /// How often `T` stands after: `A..T`
    follows: Counter<&'s UTerm>,
    /// How often `T` stands before: `T..A`
    followed_by: Counter<&'s UTerm>,
}

#[derive(Default)]
pub struct Contexts<'s> {
    map: HashMap<&'s UTerm, PairwiseFreq<'s>>,
}

impl<'s> Contexts<'s> {
    /// Record an occurrence.
    pub fn track(&mut self, left: &'s UTerm, right: &'s UTerm) {
        self.map.entry(right).or_default().follows.inc(left);
        self.map.entry(left).or_default().followed_by.inc(right);
    }

    /// The total number of cases where `term` stands on the left side of `by`: `term…by`.
    pub fn cases_term_is_followed(&self, term: &'s UTerm, by: &'s UTerm) -> usize {
        self.map.get(&term).unwrap().followed_by.get(&by)
    }

    /// Value showing how dispersive the surrounding of a term is.
    /// The term may appear many times with the same words around, which means it's a fixed expression.
    ///
    /// `0` is fixed, `1` is dispersive.
    pub fn dispersion_of(&self, term: &'s UTerm) -> (f64, f64) {
        match self.map.get(&term) {
            None => (0., 0.),
            Some(PairwiseFreq { follows: leftward, followed_by: rightward }) => (
                if leftward.is_empty() { 0. } else { leftward.distinct() as f64 / leftward.total() as f64 },
                if rightward.is_empty() { 0. } else { rightward.distinct() as f64 / rightward.total() as f64 },
            ),
        }
    }
}
