use std::collections::HashSet;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Tag {
    /// "d"
    Digit,
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
    pub fn is_numeric(word: impl AsRef<str>) -> bool {
        word.as_ref().replace(",", "").parse::<f64>().is_ok()
    }

    /// USA, USSR, but not B2C.
    pub fn is_acronym(word: &str) -> bool {
        word.chars().all(char::is_uppercase)
    }

    pub fn is_uppercase(strict_capital: bool, word: &str, is_first_word_of_sentence: bool) -> bool {
        let is_capital: fn(&str) -> bool = if strict_capital { is_strict_capitalized } else { is_capitalized };
        !is_first_word_of_sentence && is_capital(word)
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
