use unicode_segmentation::UnicodeSegmentation;

#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
pub struct PreprocessorCfg {
    #[deprecated = "not implemented yet"]
    pub ignore_urls: bool,
    #[deprecated = "not implemented yet"]
    pub expand_contractions: bool,
}

pub fn split_into_words(text: &str, _cfg: &PreprocessorCfg) -> Vec<String> {
    text.split_word_bounds()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.replace("'s", "").replace(",", ""))
        .collect()
}

pub fn split_into_sentences(text: &str, _cfg: &PreprocessorCfg) -> Vec<String> {
    text.trim()
        .replace("\n", "")
        .replace("\t", "")
        .replace("\r", "")
        .unicode_sentences()
        .map(ToString::to_string)
        .collect()
}
