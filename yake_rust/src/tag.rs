use std::collections::HashSet;

use crate::Config;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Tag {
    /// "d"
    Digit,
    /// subset of "u"
    Punctuation,
    /// "u"
    Unparsable,
    /// "a"
    Acronym,
    /// "n"
    Uppercase,
    /// "p"
    Parsable,
}

impl Tag {
    pub fn from(word: &str, is_first_word_of_sentence: bool, cfg: &Config) -> Tag {
        if Tag::is_numeric(word) {
            Tag::Digit
        } else if Tag::is_punctuation(word, &cfg.punctuation) {
            Tag::Punctuation
        } else if Tag::is_unparsable(word, &cfg.punctuation) {
            Tag::Unparsable
        } else if Tag::is_acronym(word) {
            Tag::Acronym
        } else if Tag::is_uppercase(word, is_first_word_of_sentence, cfg.strict_capital) {
            Tag::Uppercase
        } else {
            Tag::Parsable
        }
    }

    pub fn is_numeric(word: impl AsRef<str>) -> bool {
        word.as_ref().replace(",", "").parse::<f64>().is_ok()
    }

    /// USA, USSR, but not B2C.
    pub fn is_acronym(word: &str) -> bool {
        word.chars().all(char::is_uppercase)
    }

    pub fn is_uppercase(word: &str, is_first_word_of_sentence: bool, strict_capital: bool) -> bool {
        let is_capital: fn(&str) -> bool = if strict_capital { is_strict_capitalized } else { is_capitalized };
        !is_first_word_of_sentence && is_capital(word)
    }

    pub fn is_punctuation(word: &str, punctuation: &HashSet<char>) -> bool {
        word.chars().all(|c| punctuation.contains(&c))
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
