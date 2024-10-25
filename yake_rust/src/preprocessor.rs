use unicode_segmentation::UnicodeSegmentation;

#[allow(dead_code)]
pub struct Preprocessor {
    pub text: String,
    #[deprecated = "not implemented yet"]
    pub ignore_urls: bool,
    #[deprecated = "not implemented yet"]
    pub expand_contractions: bool,
}

impl Preprocessor {
    pub fn new(text: String, ignore_urls: Option<bool>, expand_contractions: Option<bool>) -> Preprocessor {
        #[allow(deprecated)]
        Preprocessor { text, ignore_urls: ignore_urls.unwrap_or(true), expand_contractions: expand_contractions.unwrap_or(true) }
    }

    pub fn split_into_words(&mut self) -> Vec<String> {
        self.text
            .split_word_bounds()
            .filter_map(|f| if f.trim().is_empty() { None } else { Some(f.trim().replace("'s", "").replace(",", "").to_string()) })
            .collect::<Vec<String>>()
    }

    pub fn split_into_sentences(&self) -> Vec<String> {
        let sents = self.text.trim().replace("\n", "").replace("\t", "").replace("\r", "");
        sents.unicode_sentences().map(|f| f.to_string()).collect::<Vec<String>>()
    }
}
