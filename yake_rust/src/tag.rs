use std::collections::HashSet;

use crate::Occurrence;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Tag {
    /// `d`
    Digit,
    /// `u`
    Unparsable,
    /// `A`
    Acronym,
    /// `N`
    Uppercase,
    Parsable,
}

impl Tag {
    pub fn is_numeric(word: impl AsRef<str>) -> bool {
        word.as_ref().replace(",", "").parse::<f64>().is_ok()
    }

    /// USA, USSR, but not B2C.
    pub fn is_acronym(word: &str) -> bool {
        word.chars().all(char::is_uppercase)
    }

    pub fn is_uppercase(strict_capital: bool, occurrence: &Occurrence) -> bool {
        let is_capital = if strict_capital { is_strict_capitalized } else { is_capitalized };
        !occurrence.is_first_word_of_sentence() && is_capital(occurrence.word)
    }

    pub fn is_unparsable(word: &str, punctuation: &HashSet<char>) -> bool {
        word_has_multiple_punctuation_symbols(word, punctuation) || {
            let has_digits = word.chars().any(|w| w.is_ascii_digit());
            let has_alphas = word.chars().any(|w| w.is_alphabetic());
            has_alphas == has_digits
        }
    }
}

/// The first symbol is uppercase.
pub fn is_capitalized(word: &str) -> bool {
    word.chars().next().is_some_and(char::is_uppercase)
}

/// Only the first symbol is uppercase.
pub fn is_strict_capitalized(word: &str) -> bool {
    let mut chars = word.chars();
    chars.next().is_some_and(char::is_uppercase) && !chars.any(char::is_uppercase)
}

fn word_has_multiple_punctuation_symbols(word: &str, punctuation: &HashSet<char>) -> bool {
    HashSet::from_iter(word.chars()).intersection(punctuation).count() > 1
}
