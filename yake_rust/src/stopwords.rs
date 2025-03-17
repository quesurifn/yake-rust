use std::ops::{Deref, DerefMut};

use crate::LTerm;

/// Contains words to be filtered out from the resulting set.
///
/// The list is used to mark potentially meaningless tokens and generally based on the _language_
/// given as input.
///
/// Tokens with fewer than three characters are also considered a stopword.
#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct StopWords {
    set: hashbrown::HashSet<LTerm>,
}

impl StopWords {
    /// Use the passed set of lowercased strings as stopwords.
    pub fn custom(lowercased: std::collections::HashSet<LTerm>) -> Self {
        Self::from(lowercased)
    }

    /// Load a predefined list of stopwords for the language given as argument.
    ///
    /// The argument is a [ISO 639 two-letter code](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes).
    /// See the [isolang](https://docs.rs/isolang/latest/isolang/index.html) crate.
    pub fn predefined(lang_iso_639_2: &str) -> Option<Self> {
        // https://github.com/LIAAD/yake/tree/0fa58cceb465162b6bd0cab7ec967edeb907fbcc/yake/StopwordsList
        // files were taken from the original repository, with extra modifications:
        // - add extra line at the end
        // - fix encoding, convert to utf8
        // - switch from CRLF to LF
        let file = match lang_iso_639_2 {
            #[cfg(feature = "ar")]
            "ar" => include_str!("stopwords/ar.txt"),
            #[cfg(feature = "bg")]
            "bg" => include_str!("stopwords/bg.txt"),
            #[cfg(feature = "br")]
            "br" => include_str!("stopwords/br.txt"),
            #[cfg(feature = "cz")]
            "cz" => include_str!("stopwords/cz.txt"),
            #[cfg(feature = "da")]
            "da" => include_str!("stopwords/da.txt"),
            #[cfg(feature = "de")]
            "de" => include_str!("stopwords/de.txt"),
            #[cfg(feature = "el")]
            "el" => include_str!("stopwords/el.txt"),
            #[cfg(feature = "en")]
            "en" => include_str!("stopwords/en.txt"),
            #[cfg(feature = "es")]
            "es" => include_str!("stopwords/es.txt"),
            #[cfg(feature = "et")]
            "et" => include_str!("stopwords/et.txt"),
            #[cfg(feature = "fa")]
            "fa" => include_str!("stopwords/fa.txt"),
            #[cfg(feature = "fi")]
            "fi" => include_str!("stopwords/fi.txt"),
            #[cfg(feature = "fr")]
            "fr" => include_str!("stopwords/fr.txt"),
            #[cfg(feature = "hi")]
            "hi" => include_str!("stopwords/hi.txt"),
            #[cfg(feature = "hr")]
            "hr" => include_str!("stopwords/hr.txt"),
            #[cfg(feature = "hu")]
            "hu" => include_str!("stopwords/hu.txt"),
            #[cfg(feature = "hy")]
            "hy" => include_str!("stopwords/hy.txt"),
            #[cfg(feature = "id")]
            "id" => include_str!("stopwords/id.txt"),
            #[cfg(feature = "it")]
            "it" => include_str!("stopwords/it.txt"),
            #[cfg(feature = "ja")]
            "ja" => include_str!("stopwords/ja.txt"),
            #[cfg(feature = "lt")]
            "lt" => include_str!("stopwords/lt.txt"),
            #[cfg(feature = "lv")]
            "lv" => include_str!("stopwords/lv.txt"),
            #[cfg(feature = "nl")]
            "nl" => include_str!("stopwords/nl.txt"),
            #[cfg(feature = "no")]
            "no" => include_str!("stopwords/no.txt"),
            #[cfg(feature = "pl")]
            "pl" => include_str!("stopwords/pl.txt"),
            #[cfg(feature = "pt")]
            "pt" => include_str!("stopwords/pt.txt"),
            #[cfg(feature = "ro")]
            "ro" => include_str!("stopwords/ro.txt"),
            #[cfg(feature = "ru")]
            "ru" => include_str!("stopwords/ru.txt"),
            #[cfg(feature = "sk")]
            "sk" => include_str!("stopwords/sk.txt"),
            #[cfg(feature = "sl")]
            "sl" => include_str!("stopwords/sl.txt"),
            #[cfg(feature = "sv")]
            "sv" => include_str!("stopwords/sv.txt"),
            #[cfg(feature = "tr")]
            "tr" => include_str!("stopwords/tr.txt"),
            #[cfg(feature = "uk")]
            "uk" => include_str!("stopwords/uk.txt"),
            #[cfg(feature = "zh")]
            "zh" => include_str!("stopwords/zh.txt"),
            _ => return None,
        };

        Some(Self { set: file.lines().map(ToOwned::to_owned).collect() })
    }
}

impl From<hashbrown::HashSet<LTerm>> for StopWords {
    fn from(lowercased: hashbrown::HashSet<LTerm>) -> Self {
        Self { set: lowercased.into_iter().collect() }
    }
}

impl From<std::collections::HashSet<LTerm>> for StopWords {
    fn from(lowercased: std::collections::HashSet<LTerm>) -> Self {
        Self { set: lowercased.into_iter().collect() }
    }
}

impl Deref for StopWords {
    type Target = hashbrown::HashSet<LTerm>;

    fn deref(&self) -> &Self::Target {
        &self.set
    }
}

impl<T> AsRef<T> for StopWords
where
    T: ?Sized,
    <StopWords as Deref>::Target: AsRef<T>,
{
    fn as_ref(&self) -> &T {
        self.deref().as_ref()
    }
}

impl DerefMut for StopWords {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.set
    }
}
