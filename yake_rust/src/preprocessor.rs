use segtok::segmenter::split_multi;
use segtok::tokenizer::{split_contractions, web_tokenizer};

pub fn split_into_words(text: &str) -> Vec<String> {
    split_contractions(web_tokenizer(text))
        .into_iter()
        .filter(|word| !(word.is_empty() || word.len() > 1 && word.starts_with("'")))
        .collect()
}

pub fn split_into_sentences(text: &str) -> Vec<String> {
    split_multi(text, Default::default()).into_iter().filter(|span| !span.is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_hyphenated_sentence_into_words() {
        let text = "Truly high-tech!";
        let expected = ["Truly", "high-tech", "!"];
        let actual = split_into_words(&text);
        assert_eq!(actual, expected);
    }

    #[test]
    fn split_sentences() {
        let text = "One smartwatch. One phone. Many phones.";
        let expected = ["One smartwatch.", "One phone.", "Many phones."];
        let actual = split_into_sentences(&text);
        assert_eq!(actual, expected);
    }

    #[test]
    fn split_tabbed_multiline_text_into_sentences() {
        let text = "This is your weekly newsletter! \
            Hundreds of great deals - everything from men's fashion \
            to high-tech drones!";
        let expected = [
            "This is your weekly newsletter!",
            "Hundreds of great deals - everything from men's fashion to high-tech drones!",
        ];
        let actual = split_into_sentences(&text);
        assert_eq!(actual, expected);
    }
}
