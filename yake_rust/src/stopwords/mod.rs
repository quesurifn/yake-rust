use std::collections::HashSet;
use std::ops::{Deref, DerefMut};

use crate::LTerm;

/// List of lowercased words to be filtered out from the text.
///
/// The list is used to mark potentially meaningless tokens and generally based on the language
/// given as input. Tokens with fewer than three characters are also considered a stopword.
#[derive(Debug, Default, Clone)]
pub struct StopWords {
    set: HashSet<LTerm>,
}

impl StopWords {
    /// Use the passed set of lowercased strings as stopwords.
    pub fn custom(lowercased: HashSet<LTerm>) -> Self {
        StopWords { set: lowercased }
    }

    /// Load a predefined list of stopwords for the language given as argument.
    pub fn predefined(lang: &str) -> Result<Self, String> {
        // https://github.com/LIAAD/yake/tree/0fa58cceb465162b6bd0cab7ec967edeb907fbcc/yake/StopwordsList
        // files were taken from the original repository, with extra modifications:
        // - add extra line at the end
        // - fix encoding, convert to utf8
        // - switch from CRLF to LF
        let file = match lang {
            #[cfg(feature = "ar")]
            "ar" => include_str!("ar.txt"),
            #[cfg(feature = "bg")]
            "bg" => include_str!("bg.txt"),
            #[cfg(feature = "br")]
            "br" => include_str!("br.txt"),
            #[cfg(feature = "cz")]
            "cz" => include_str!("cz.txt"),
            #[cfg(feature = "da")]
            "da" => include_str!("da.txt"),
            #[cfg(feature = "de")]
            "de" => include_str!("de.txt"),
            #[cfg(feature = "el")]
            "el" => include_str!("el.txt"),
            #[cfg(feature = "en")]
            "en" => include_str!("en.txt"),
            #[cfg(feature = "es")]
            "es" => include_str!("es.txt"),
            #[cfg(feature = "et")]
            "et" => include_str!("et.txt"),
            #[cfg(feature = "fa")]
            "fa" => include_str!("fa.txt"),
            #[cfg(feature = "fi")]
            "fi" => include_str!("fi.txt"),
            #[cfg(feature = "fr")]
            "fr" => include_str!("fr.txt"),
            #[cfg(feature = "hi")]
            "hi" => include_str!("hi.txt"),
            #[cfg(feature = "hr")]
            "hr" => include_str!("hr.txt"),
            #[cfg(feature = "hu")]
            "hu" => include_str!("hu.txt"),
            #[cfg(feature = "hy")]
            "hy" => include_str!("hy.txt"),
            #[cfg(feature = "id")]
            "id" => include_str!("id.txt"),
            #[cfg(feature = "it")]
            "it" => include_str!("it.txt"),
            #[cfg(feature = "ja")]
            "ja" => include_str!("ja.txt"),
            #[cfg(feature = "lt")]
            "lt" => include_str!("lt.txt"),
            #[cfg(feature = "lv")]
            "lv" => include_str!("lv.txt"),
            #[cfg(feature = "nl")]
            "nl" => include_str!("nl.txt"),
            #[cfg(feature = "no")]
            "no" => include_str!("no.txt"),
            #[cfg(feature = "pl")]
            "pl" => include_str!("pl.txt"),
            #[cfg(feature = "pt")]
            "pt" => include_str!("pt.txt"),
            #[cfg(feature = "ro")]
            "ro" => include_str!("ro.txt"),
            #[cfg(feature = "ru")]
            "ru" => include_str!("ru.txt"),
            #[cfg(feature = "sk")]
            "sk" => include_str!("sk.txt"),
            #[cfg(feature = "sl")]
            "sl" => include_str!("sl.txt"),
            #[cfg(feature = "sv")]
            "sv" => include_str!("sv.txt"),
            #[cfg(feature = "tr")]
            "tr" => include_str!("tr.txt"),
            #[cfg(feature = "uk")]
            "uk" => include_str!("uk.txt"),
            #[cfg(feature = "zh")]
            "zh" => include_str!("zh.txt"),
            _ => return Err(format!("Could not find language {}, did you enable this feature?", lang)),
        };

        Ok(Self { set: file.lines().map(ToOwned::to_owned).collect() })
    }
}

impl StopWords {
    pub fn contains(&self, word: &str) -> bool {
        self.set.contains(word)
    }
}

impl Deref for StopWords {
    type Target = HashSet<LTerm>;

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
