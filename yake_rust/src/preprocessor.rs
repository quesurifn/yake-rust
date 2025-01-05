use unicode_segmentation::UnicodeSegmentation;

pub fn split_into_words(text: &str) -> Vec<String> {
    text.split_word_bounds()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.replace("'s", "").replace(",", ""))
        .collect()
}

pub fn split_into_sentences(text: &str) -> Vec<String> {
    text.trim()
        .replace("\n", "")
        .replace("\t", "")
        .replace("\r", "")
        .unicode_sentences()
        .map(ToString::to_string)
        .collect()
}
