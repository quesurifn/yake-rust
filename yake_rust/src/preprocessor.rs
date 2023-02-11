use unicode_segmentation::UnicodeSegmentation;

pub struct Preprocessor {
    pub text: String,
    pub ignore_urls: bool,
    pub expand_contractions: bool,
}

impl Preprocessor {
    pub fn new(text: String, ignore_urls: Option<bool>, expand_contractions: Option<bool>) -> Preprocessor {
        let default_ignore_urls = ignore_urls.unwrap_or(true);
        let default_expand_contractions = expand_contractions.unwrap_or(true);
        Preprocessor {
            text,
            ignore_urls: default_ignore_urls,
            expand_contractions: default_expand_contractions,
        }
    }

    pub fn split_into_words(&mut self) -> Vec<String> {
        self.text.split_word_bounds().filter_map(|f| {
            if f.trim().is_empty() {
                None
            } else {
                Some(f.trim().replace("'s", "").replace(",", "").to_string())
            }
        }).collect::<Vec<String>>()
    }
    
    pub fn split_into_sentences(&self) -> Vec<String> {
        let sents = self.text.trim().replace("\n", "").replace("\t", "").replace("\r", "");
        sents.unicode_sentences().map(|f| f.to_string()).collect::<Vec<String>>()
    }
}