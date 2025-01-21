#![allow(clippy::len_zero)]
#![allow(clippy::type_complexity)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::iter::FromIterator;

use indexmap::{IndexMap, IndexSet};
use plural_helper::PluralHelper;
use preprocessor::{split_into_sentences, split_into_words};
use stats::{mean, median, stddev};

use crate::context::Contexts;
use crate::levenshtein::levenshtein_ratio;
pub use crate::stopwords::StopWords;

mod context;
mod counter;
mod levenshtein;
mod plural_helper;
mod preprocessor;
mod stopwords;

/// String from the original text
type RawString = String;

/// Lowercased string
type LTerm = String;

/// Lowercased string without punctuation symbols in single form
type UTerm = String;

type Sentences = Vec<Sentence>;
type Candidates<'s> = IndexMap<&'s [LTerm], Candidate<'s>>;
type Features<'s> = HashMap<&'s LTerm, TermStats>;
type Words<'s> = HashMap<&'s UTerm, Vec<Occurrence<'s>>>;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Tag {
    Digit,
    Unparsable,
    Acronym,
    Uppercase,
    Parsable,
}

#[derive(PartialEq, Eq, Hash, Debug)]
struct Occurrence<'sentence> {
    /// Ordinal number of the word in the source text after splitting into sentences
    pub shift_offset: usize,
    /// The total number of words in all previous sentences.
    pub shift: usize,
    /// Index (0..) of sentence where the term occur
    pub idx: usize,
    /// The word itself
    pub word: &'sentence String,
}

impl<'s> Occurrence<'s> {
    /// USA, USSR, but not B2C.
    fn _is_acronym(&self) -> bool {
        self.word.len() > 1 && self.word.chars().all(char::is_uppercase)
    }

    /// The first symbol is uppercase.
    fn is_uppercased(&self) -> bool {
        self.word.chars().next().is_some_and(char::is_uppercase)
    }

    /// Only the first symbol is uppercase.
    fn is_capitalized(&self) -> bool {
        let mut chars = self.word.chars();
        chars.next().is_some_and(char::is_uppercase) && !chars.any(char::is_uppercase)
    }

    fn is_first_word(&self) -> bool {
        self.shift == self.shift_offset
    }
}

#[derive(Debug, Default)]
struct TermStats {
    is_stopword: bool,
    /// Term frequency. The total number of occurrences in the text.
    tf: f64,
    /// The number of times this candidate term is marked as an acronym (=all uppercase).
    tf_a: f64,
    /// The number of occurrences of this candidate term starting with an uppercase letter.
    tf_n: f64,
    /// Term casing heuristic.
    casing: f64,
    /// Term position heuristic
    position: f64,
    /// Normalized term frequency heuristic
    frequency: f64,
    /// Term relatedness to context
    relatedness: f64,
    /// Term's different sentences heuristic
    sentences: f64,
    /// Importance score. The less, the better
    score: f64,
}

#[derive(PartialEq, Clone, Debug)]
pub struct ResultItem {
    pub raw: String,
    pub keyword: LTerm,
    pub score: f64,
}

impl PartialEq<(&str, &str, f64)> for ResultItem {
    fn eq(&self, (raw, keyword, score): &(&str, &str, f64)) -> bool {
        self.raw.eq(raw) && self.keyword.eq(keyword) && self.score.eq(score)
    }
}

#[derive(Debug, Clone)]
struct Sentence {
    pub words: Vec<RawString>,
    pub is_punctuation: Vec<bool>,
    pub lc_terms: Vec<LTerm>,
    pub uq_terms: Vec<UTerm>,
}

/// N-gram, a sequence of N terms.
#[derive(Debug, Default, Clone)]
struct Candidate<'s> {
    pub occurrences: Vec<&'s [RawString]>,
    pub lc_terms: &'s [LTerm],
    pub uq_terms: &'s [UTerm],
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct Config {
    /// The number of n-grams.
    ///
    /// _n-gram_ is a contiguous sequence of _n_ words occurring in the text.
    pub ngrams: usize,
    /// List of punctuation symbols.
    ///
    /// They are known as _exclude chars_ in the original implementation.
    pub punctuation: HashSet<char>,
    pub window_size: usize,
    pub remove_duplicates: bool,
    /// A threshold in range 0..1.
    pub deduplication_threshold: f64,
    /// When `true`, calculate _term casing_ metric by counting capitalized terms _without_
    /// intermediate uppercase letters. Thus, `Paypal` is counted while `PayPal` is not.
    ///
    /// The [original implementation](https://github.com/LIAAD/) sticks with `true`.
    pub strict_capital: bool,

    /// When `true`, key phrases are allowed to have only alphanumeric characters and hyphen.
    pub only_alphanumeric_and_hyphen: bool,
    /// Key phrases can't be too short, less than `minimum_chars` in total.
    pub minimum_chars: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            punctuation: r##"!"#$%&'()*+,-./:,<=>?@[\]^_`{|}~"##.chars().collect(),
            window_size: 1,
            deduplication_threshold: 0.9,
            ngrams: 3,
            remove_duplicates: true,
            strict_capital: true,
            only_alphanumeric_and_hyphen: false,
            minimum_chars: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Yake {
    config: Config,
    stop_words: StopWords,
}

impl Yake {
    pub fn new(stop_words: StopWords, config: Config) -> Yake {
        Self { config, stop_words }
    }

    pub fn get_n_best(&self, text: &str, n: Option<usize>) -> Vec<ResultItem> {
        let sentences = self.preprocess_text(text);

        let context = self.build_context(&sentences);
        let vocabulary = self.build_vocabulary(&sentences);
        let features = self.extract_features(&context, vocabulary, &sentences);

        let mut ngrams: Candidates = self.ngram_selection(self.config.ngrams, &sentences);
        Yake::candidate_weighting(features, &context, &mut ngrams);

        let mut results = ngrams
            .into_iter()
            .map(|(_, candidate)| {
                let raw = candidate.occurrences[0].join(" ");
                let keyword = candidate.lc_terms.join(" ");
                ResultItem { raw, keyword, score: candidate.score }
            })
            .collect::<Vec<_>>();

        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

        let n = n.unwrap_or(usize::MAX);
        if self.config.remove_duplicates {
            self.remove_duplicates(results, n)
        } else {
            results.truncate(n);
            results
        }
    }

    fn get_unique_term(&self, word: &str) -> UTerm {
        word.to_single().to_lowercase()
    }

    fn is_stopword(&self, lc_term: &LTerm) -> bool {
        // todo: optimize by iterating the smallest set or with a trie
        self.stop_words.contains(lc_term)
            || self.stop_words.contains(lc_term.to_single())
            // having less than 3 non-punctuation symbols is typical for stop words
            || lc_term.to_single().chars().filter(|ch| !self.config.punctuation.contains(ch)).count() < 3
    }

    pub fn contains_stopword(&self, words: &HashSet<&LTerm>) -> bool {
        words.iter().any(|w| self.is_stopword(w))
    }

    fn remove_duplicates(&self, results: Vec<ResultItem>, n: usize) -> Vec<ResultItem> {
        let mut unique: Vec<ResultItem> = Vec::new();

        for res in results {
            if unique.len() >= n {
                break;
            }

            let is_duplicate = unique
                .iter()
                .any(|it| levenshtein_ratio(&it.keyword, &res.keyword) > self.config.deduplication_threshold);

            if !is_duplicate {
                unique.push(res);
            }
        }

        unique
    }

    fn preprocess_text(&self, text: &str) -> Sentences {
        split_into_sentences(text)
            .into_iter()
            .map(|sentence| {
                let words = split_into_words(&sentence);
                let lc_words = words.iter().map(|w| w.to_lowercase()).collect::<Vec<LTerm>>();
                let uq_terms = lc_words.iter().map(|w| self.get_unique_term(w)).collect();
                let is_punctuation = words.iter().map(|w| self.word_is_punctuation(w)).collect();
                Sentence { words, lc_terms: lc_words, uq_terms, is_punctuation }
            })
            .collect()
    }

    fn build_vocabulary<'s>(&self, sentences: &'s [Sentence]) -> Words<'s> {
        let mut words = Words::new();

        for (idx, sentence) in sentences.iter().enumerate() {
            let shift = sentences[0..idx].iter().map(|s| s.words.len()).sum::<usize>();

            for (w_idx, ((word, is_punctuation), index)) in
                sentence.words.iter().zip(&sentence.is_punctuation).zip(&sentence.uq_terms).enumerate()
            {
                if !word.is_empty() && !is_punctuation {
                    let occurrence = Occurrence { shift_offset: shift + w_idx, idx, word, shift };
                    words.entry(index).or_default().push(occurrence)
                }
            }
        }

        words
    }

    /// Builds co-occurrence matrix containing the co-occurrences between
    /// a given term and its predecessor AND a given term and its subsequent term,
    /// found within a window of a given size.
    fn build_context<'s>(&self, sentences: &'s [Sentence]) -> Contexts<'s> {
        let mut ctx = Contexts::default();

        for sentence in sentences {
            let mut window: VecDeque<(&String, &UTerm)> = VecDeque::with_capacity(self.config.window_size + 1);

            for ((word, term), &is_punctuation) in
                sentence.words.iter().zip(&sentence.uq_terms).zip(&sentence.is_punctuation)
            {
                if is_punctuation {
                    window.clear();
                    continue;
                }

                // Do not store in contexts in any way if the word (not the unique term) is tagged "d" or "u"
                if !self.is_d_tagged(word) && !self.is_u_tagged(word) {
                    for &(left_word, left_uterm) in window.iter() {
                        if self.is_d_tagged(left_word) || self.is_u_tagged(left_word) {
                            continue;
                        }

                        ctx.track(left_uterm, term);
                    }
                }

                if window.len() == self.config.window_size {
                    window.pop_front();
                }
                window.push_back((word, term));
            }
        }

        ctx
    }

    fn is_d_tagged(&self, word: &str) -> bool {
        word.replace(",", "").parse::<f64>().is_ok()
    }

    fn is_u_tagged(&self, word: &str) -> bool {
        self.word_has_multiple_punctuation_symbols(word) || {
            let nr_of_digits = word.chars().filter(|w| w.is_ascii_digit()).count();
            let nr_of_alphas = word.chars().filter(|w| w.is_alphabetic()).count();
            (nr_of_alphas > 0 && nr_of_digits > 0) || (nr_of_alphas == 0 && nr_of_digits == 0)
        }
    }

    fn is_a_tagged(&self, word: &str) -> bool {
        word.chars().all(char::is_uppercase)
    }

    fn is_n_tagged(&self, occurrence: &Occurrence) -> bool {
        let is_capital =
            if self.config.strict_capital { Occurrence::is_capitalized } else { Occurrence::is_uppercased };
        is_capital(occurrence) && !occurrence.is_first_word()
    }

    fn get_tag(&self, occurrence: &Occurrence) -> Tag {
        if self.is_d_tagged(occurrence.word) {
            Tag::Digit
        } else if self.is_u_tagged(occurrence.word) {
            Tag::Unparsable
        } else if self.is_a_tagged(occurrence.word) {
            Tag::Acronym
        } else if self.is_n_tagged(occurrence) {
            Tag::Uppercase
        } else {
            Tag::Parsable
        }
    }

    /// Computes local statistic features that extract informative content within the text
    /// to calculate the importance of single terms.
    fn extract_features<'s>(&self, ctx: &Contexts, words: Words<'s>, sentences: &'s Sentences) -> Features<'s> {
        let tf = words.values().map(Vec::len);

        let words_nsw: HashMap<&UTerm, usize> = sentences
            .iter()
            .flat_map(|sentence| sentence.lc_terms.iter().zip(&sentence.uq_terms).zip(&sentence.is_punctuation))
            .filter(|&((lc_term, _), is_punct)| !is_punct && !self.is_stopword(lc_term))
            .map(|((_, u_term), _)| {
                let occurrences = words.get(u_term).unwrap().len();
                (u_term, occurrences)
            })
            .collect();

        let tf_nsw = words_nsw.values().map(|x| *x as f64);
        let std_tf = stddev(tf_nsw.clone());
        let mean_tf = mean(tf_nsw);
        let max_tf = tf.max().unwrap() as f64;

        let mut features = Features::new();

        let candidate_words: IndexSet<_> = sentences
            .iter()
            .flat_map(|sentence| sentence.lc_terms.iter().zip(&sentence.uq_terms).zip(&sentence.is_punctuation))
            .filter(|&(_, is_punct)| !is_punct)
            .map(|(pair, _)| pair)
            .collect();

        for (lc_term, u_term) in candidate_words {
            let occurrences = words.get(u_term).unwrap();

            let mut stats = TermStats {
                is_stopword: self.is_stopword(lc_term),
                tf: occurrences.len() as f64,
                ..Default::default()
            };

            {
                // todo: revert back to the code from 5a6e4747, as tags change nothing
                let tags: Vec<Tag> = occurrences.iter().map(|occ| self.get_tag(occ)).collect();
                // We consider the occurrence of acronyms through a heuristic, where all the letters of the word are capitals.
                stats.tf_a = tags.iter().filter(|&tag| *tag == Tag::Acronym).count() as f64;
                // We give extra attention to any term beginning with a capital letter (excluding the beginning of sentences).
                stats.tf_n = tags.iter().filter(|&tag| *tag == Tag::Uppercase).count() as f64;

                // The casing aspect of a term is an important feature when considering the extraction
                // of keywords. The underlying rationale is that uppercase terms tend to be more
                // relevant than lowercase ones.
                stats.casing = stats.tf_a.max(stats.tf_n);

                // The more often the candidate term occurs with a capital letter, the more important
                // it is considered. This means that a candidate term that occurs with a capital letter
                // ten times within ten occurrences will be given a higher value (4.34) than a candidate
                // term that occurs with a capital letter five times within five occurrences (3.10).
                stats.casing /= 1.0 + stats.tf.ln(); // todo: no 1+ in the paper
            }

            {
                // Another indicator of the importance of a candidate term is its position.
                // The rationale is that relevant keywords tend to appear at the very beginning
                // of a document, whereas words occurring in the middle or at the end of a document
                // tend to be less important.
                //
                // This is particularly evident for both news articles and scientific texts,
                // which tend to concentrate a high degree of important
                // keywords at the top of the text (e.g., in the introduction or abstract).
                //
                // Like Florescu and Caragea, who posit that models that take into account the positions
                // of terms perform better than those that only use the first position or no position
                // at all, we also consider a term’s position to be an important feature. However,
                // unlike their model, we do not consider the positions of the terms,
                // but of the sentences in which the terms occur.
                //
                // Our assumption is that terms that occur in the early
                // sentences of a text should be more highly valued than terms that appear later. Thus,
                // instead of considering a uniform distribution of terms, our model assigns higher
                // scores to terms found in early sentences. todo: optimize space, dedup sorted indexes
                let sentence_ids: HashSet<_> = occurrences.iter().map(|o| o.idx).collect();
                // When the candidate term only appears in the first sentence, the median function
                // can return a value of 0. To guarantee position > 0, a constant 3 > e=2.71 is added.
                stats.position = 3.0 + median(sentence_ids.into_iter()).unwrap();
                // A double log is applied in order to smooth the difference between terms that occur
                // with a large median difference.
                stats.position = stats.position.ln().ln();
            }

            {
                // The higher the frequency of a candidate term, the greater its importance.
                stats.frequency = stats.tf;
                // To prevent a bias towards high frequency in long documents, we balance it.
                // The mean and the standard deviation is calculated from non-stopwords terms,
                // as stopwords usually have high frequencies.
                stats.frequency /= mean_tf + std_tf;
            }

            {
                let (dl, dr) = ctx.dispersion_of(u_term);
                stats.relatedness = 1.0 + (dr + dl) * (stats.tf / max_tf);
            }

            {
                // Candidates which appear in many different sentences have a higher probability
                // of being important.
                let distinct = occurrences.iter().map(|o| o.idx).collect::<HashSet<_>>();
                stats.sentences = distinct.len() as f64 / sentences.len() as f64;
            }

            stats.score = (stats.relatedness * stats.position)
                / (stats.casing + (stats.frequency / stats.relatedness) + (stats.sentences / stats.relatedness));

            features.insert(lc_term, stats);
        }

        features
    }

    fn candidate_weighting<'s>(features: Features<'s>, ctx: &Contexts<'s>, candidates: &mut Candidates<'s>) {
        for candidate in candidates.values_mut() {
            let lc_terms = candidate.lc_terms;
            let uq_terms = candidate.uq_terms;
            {
                let mut prod_ = 1.0;
                let mut sum_ = 0.0;

                for (j, (lc, uq)) in lc_terms.iter().zip(uq_terms).enumerate() {
                    let Some(stats) = features.get(lc) else { continue };
                    if stats.is_stopword {
                        let mut prob_prev = 0.0;
                        let mut prob_succ = 0.0;
                        if 0 < j {
                            // Not the first term
                            // #previous term occurring before this one / #previous term
                            let prev_uq = uq_terms.get(j - 1).unwrap();
                            let prev_lc = lc_terms.get(j - 1).unwrap();
                            prob_prev =
                                ctx.cases_term_is_followed(&prev_uq, &uq) as f64 / features.get(&prev_lc).unwrap().tf;
                        }
                        if j < uq_terms.len() {
                            // Not the last term
                            // #next term occurring after this one / #next term
                            let next_uq = uq_terms.get(j + 1).unwrap();
                            let next_lc = lc_terms.get(j + 1).unwrap();
                            prob_succ =
                                ctx.cases_term_is_followed(&uq, &next_uq) as f64 / features.get(&next_lc).unwrap().tf;
                            // fixme: Probability P(T[i+1] | T[i]) is weird.
                            //        Why divide by Fr(T[i]) at first, but by Fr(T[i+1]) at second?
                        }

                        let prob = prob_prev * prob_succ;
                        prod_ *= 1.0 + (1.0 - prob);
                        sum_ -= 1.0 - prob;
                    } else {
                        prod_ *= stats.score;
                        sum_ += stats.score;
                    }
                }
                if sum_ == -1.0 {
                    sum_ = 0.999999999;
                }

                let tf = candidate.occurrences.len() as f64;
                candidate.score = prod_ / (tf * (1.0 + sum_));
            }
        }
    }

    fn is_candidate(&self, lc_terms: &[LTerm]) -> bool {
        let lc_words: HashSet<&LTerm> = HashSet::from_iter(lc_terms);

        let has_float = || lc_words.iter().any(|&w| self.is_d_tagged(w));
        let has_stop_word = || self.is_stopword(&lc_terms[0]) || self.is_stopword(lc_terms.last().unwrap());
        let has_unparsable = || lc_words.iter().any(|&w| self.is_u_tagged(w));
        let not_enough_symbols =
            || lc_terms.iter().map(|w| w.chars().count()).sum::<usize>() < self.config.minimum_chars;
        let has_non_alphanumeric =
            || self.config.only_alphanumeric_and_hyphen && !lc_words.iter().all(word_is_alphanumeric_and_hyphen);

        !{ has_float() || has_stop_word() || has_unparsable() || not_enough_symbols() || has_non_alphanumeric() }
    }

    fn ngram_selection<'s>(&self, n: usize, sentences: &'s Sentences) -> Candidates<'s> {
        let mut candidates = Candidates::new();
        let mut ignored = HashSet::new();

        for sentence in sentences.iter() {
            let length = sentence.words.len();

            for j in 0..length {
                for k in (j + 1..length + 1).take(n) {
                    if (j..k).is_empty() {
                        continue;
                    }

                    let lc_terms = &sentence.lc_terms[j..k];

                    if ignored.contains(lc_terms) {
                        continue;
                    }
                    if !self.is_candidate(lc_terms) {
                        ignored.insert(lc_terms);
                        continue;
                    }

                    let candidate = candidates.entry(lc_terms).or_default();
                    candidate.lc_terms = lc_terms;
                    candidate.occurrences.push(&sentence.words[j..k]);
                    candidate.uq_terms = &sentence.uq_terms[j..k];
                }
            }
        }

        candidates
    }

    fn word_has_multiple_punctuation_symbols(&self, word: impl AsRef<str>) -> bool {
        HashSet::from_iter(word.as_ref().chars()).intersection(&self.config.punctuation).count() > 1
    }

    fn word_is_punctuation(&self, word: impl AsRef<str>) -> bool {
        HashSet::from_iter(word.as_ref().chars()).is_subset(&self.config.punctuation)
    }
}

fn word_is_alphanumeric_and_hyphen(word: impl AsRef<str>) -> bool {
    word.as_ref().chars().all(|ch| ch.is_alphanumeric() || ch == '-')
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn short() {
        let text = "this is a keyword";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(1));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [("keyword", "keyword", 0.1583)];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn order() {
        // Verifies that order of keywords with the same score is preserved.
        // If not, this test becomes unstable.
        let text = "Machine learning";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 1, ..Default::default() }).get_n_best(text, Some(3));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [("Machine", "machine", 0.1583), ("learning", "learning", 0.1583)];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn laptop() {
        let text = "Do you need an Apple laptop?";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 1, ..Default::default() }).get_n_best(text, Some(2));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [("Apple", "apple", 0.1448), ("laptop", "laptop", 0.1583)];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn headphones() {
        let text = "Do you like headphones? \
        Starting this Saturday, we will be kicking off a huge sale of headphones! \
        If you need headphones, we've got you covered!";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 1, ..Default::default() }).get_n_best(text, Some(3));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected =
            [("headphones", "headphones", 0.1141), ("Saturday", "saturday", 0.2111), ("Starting", "starting", 0.4096)];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn multi_ngram() {
        let text = "I will give you a great deal if you just read this!";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 2, ..Default::default() }).get_n_best(text, Some(1));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [("great deal", "great deal", 0.0257)];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn singular() {
        let text = "One smartwatch. One phone. Many phone."; // Weird grammar; to compare with the "plural" test
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 1, ..Default::default() }).get_n_best(text, Some(2));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [("smartwatch", "smartwatch", 0.2025), ("phone", "phone", 0.2474)];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn plural() {
        let text = "One smartwatch. One phone. Many phones.";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 1, ..Default::default() }).get_n_best(text, Some(3));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [("smartwatch", "smartwatch", 0.2025), ("phone", "phone", 0.4949), ("phones", "phones", 0.4949)];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn non_hyphenated() {
        let text = "Truly high tech!"; // For comparison with the "hyphenated" test
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 2, ..Default::default() }).get_n_best(text, Some(1));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [("high tech", "high tech", 0.0494)];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn hyphenated() {
        let text = "Truly high-tech!";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 2, ..Default::default() }).get_n_best(text, Some(1));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [("high-tech", "high-tech", 0.1583)];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn weekly_newsletter_short() {
        let text = "This is your weekly newsletter!";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 2, ..Default::default() }).get_n_best(text, Some(3));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("weekly newsletter", "weekly newsletter", 0.0494),
            ("newsletter", "newsletter", 0.1583),
            ("weekly", "weekly", 0.2974),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn weekly_newsletter_long() {
        let text = "This is your weekly newsletter! \
            Hundreds of great deals - everything from men's fashion \
            to high-tech drones!";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 2, ..Default::default() }).get_n_best(text, Some(5));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("weekly newsletter", "weekly newsletter", 0.0780),
            ("newsletter", "newsletter", 0.2005),
            ("weekly", "weekly", 0.3607),
            ("great deals", "great deals", 0.4456),
            ("high-tech drones", "high-tech drones", 0.4456),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn weekly_newsletter_long_with_paragraphs() {
        let text = "This is your weekly newsletter!\n\n \
            \tHundreds of great deals - everything from men's fashion \n\
            to high-tech drones!";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 2, ..Default::default() }).get_n_best(text, Some(5));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("weekly newsletter", "weekly newsletter", 0.0780),
            ("newsletter", "newsletter", 0.2005),
            ("weekly", "weekly", 0.3607),
            ("great deals", "great deals", 0.4456),
            ("high-tech drones", "high-tech drones", 0.4456),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn composite_recurring_words_and_bigger_window() {
        let text = "Machine learning is a growing field. Few research fields grow as much as machine learning grows.";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual =
            Yake::new(stopwords, Config { ngrams: 2, window_size: 2, ..Default::default() }).get_n_best(text, Some(5));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Machine learning", "machine learning", 0.1346),
            ("growing field", "growing field", 0.1672),
            ("learning", "learning", 0.2265),
            ("Machine", "machine", 0.2341),
            ("growing", "growing", 0.2799),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn composite_recurring_words_near_numbers() {
        let text = "I buy 100 yellow bananas every day. Every night I eat bananas - all but 5 bananas.";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 2, ..Default::default() }).get_n_best(text, Some(3));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected =
            [("yellow bananas", "yellow bananas", 0.0682), ("buy", "buy", 0.1428), ("yellow", "yellow", 0.1428)];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn composite_recurring_words_near_spelled_out_numbers() {
        // For comparison with "composite_recurring_words_near_numbers" to see if numbers cause
        let text = "I buy a hundred yellow bananas every day. Every night I eat bananas - all but five bananas.";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 2, ..Default::default() }).get_n_best(text, Some(3));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("hundred yellow", "hundred yellow", 0.0446),
            ("yellow bananas", "yellow bananas", 0.1017),
            ("day", "day", 0.1428),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn with_stopword_in_the_middle() {
        let text = "Game of Thrones";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual =
            Yake::new(stopwords, Config { remove_duplicates: false, ..Config::default() }).get_n_best(text, Some(1));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [("Game of Thrones", "game of thrones", 0.01380)];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn google_sample_single_ngram() {
        let text = include_str!("test_google.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config { ngrams: 1, ..Default::default() }).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Google", "google", 0.0251),
            ("Kaggle", "kaggle", 0.0273),
            ("data", "data", 0.08),
            ("science", "science", 0.0983),
            ("platform", "platform", 0.124),
            ("service", "service", 0.1316),
            ("acquiring", "acquiring", 0.1511),
            ("learning", "learning", 0.1621),
            ("Goldbloom", "goldbloom", 0.1625),
            ("machine", "machine", 0.1672),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn google_sample_defaults() {
        let text = include_str!("test_google.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Google", "google", 0.0251),
            ("Kaggle", "kaggle", 0.0273),
            ("CEO Anthony Goldbloom", "ceo anthony goldbloom", 0.0483),
            ("data science", "data science", 0.055),
            ("acquiring data science", "acquiring data science", 0.0603),
            ("Google Cloud Platform", "google cloud platform", 0.0746),
            ("data", "data", 0.08),
            ("San Francisco", "san francisco", 0.0914),
            ("Anthony Goldbloom declined", "anthony goldbloom declined", 0.0974),
            ("science", "science", 0.0983),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn gitter_sample_defaults() {
        let text = include_str!("test_gitter.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Gitter", "gitter", 0.0190),
            ("GitLab", "gitlab", 0.0478),
            ("acquires software chat", "acquires software chat", 0.0479),
            ("chat startup Gitter", "chat startup gitter", 0.0512),
            ("software chat startup", "software chat startup", 0.0612),
            ("Gitter chat", "gitter chat", 0.0684),
            ("GitLab acquires software", "gitlab acquires software", 0.0685),
            ("startup", "startup", 0.0783),
            ("software", "software", 0.0879),
            ("code", "code", 0.0879),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn genius_sample_defaults() {
        let text = include_str!("test_genius.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Genius", "genius", 0.0261),
            ("company", "company", 0.0263),
            ("Genius quietly laid", "genius quietly laid", 0.027),
            ("company quietly laid", "company quietly laid", 0.0392),
            ("media company", "media company", 0.0404),
            ("Lehman", "lehman", 0.0412),
            ("quietly laid", "quietly laid", 0.0583),
            ("Tom Lehman told", "tom lehman told", 0.0603),
            ("video", "video", 0.0650),
            ("co-founder Tom Lehman", "co-founder tom lehman", 0.0669),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn german_sample_defaults() {
        let text = include_str!("test_german.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("de").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Vereinigten Staaten", "vereinigten staaten", 0.0152), // LIAAD REFERENCE: 0.151
            ("Präsidenten Donald Trump", "präsidenten donald trump", 0.0182),
            ("Donald Trump", "donald trump", 0.0211), // LIAAD REFERENCE: 0.21
            ("trifft Donald Trump", "trifft donald trump", 0.0231), // LIAAD REFERENCE: 0.23
            ("Trump", "trump", 0.0240),
            ("Trumps Finanzminister Steven", "trumps finanzminister steven", 0.0243),
            ("Kanzlerin Angela Merkel", "kanzlerin angela merkel", 0.0275), // LIAAD REFERENCE: 0.273
            ("deutsche Kanzlerin Angela", "deutsche kanzlerin angela", 0.0316), // LIAAD REFERENCE: 0.314
            ("Merkel trifft Donald", "merkel trifft donald", 0.0353),       // LIAAD REFERENCE: 0.351
            ("Exportnation Deutschland", "exportnation deutschland", 0.038), // LIAAD REFERENCE: 0.0379
        ];

        // REASONS FOR DISCREPANCY:
        // - The text contains both "bereit" ("ready") and "bereits" ("already").
        //   While "bereits" is a stopword, "bereit" is not.
        //   LIAAD/yake keeps track of whether a term is a stopword or not
        //   in a key-value mapping, where the key is the term, lowercase, plural-normalized.
        //   (Note that the plural normalization techique used is rarely effective in German.)
        //   Since "bereits" occurs before "bereit" in the text, LIAAD/yake sees it,
        //   recognizes it is a stopword, and stores it under the key "bereit". Later,
        //   when it encounters "bereit" (NOT a stopword), it already has that key in its
        //   mapping so it looks it up and finds that it is a keyword (which it is not).
        //   Meanwhile, yake-rust does not have such a key-value store, so it correctly
        //   recognizes "bereits" as a stopword and "bereit" as a non-stopword. The extra
        //   inclusion of "bereit" in the non-stopwords affects the TF statistics and thus
        //   the frequency contribution to the weights, leading to slightly different scores.

        assert_eq!(actual, expected);
    }

    #[test]
    fn dutch_sample_defaults() {
        let text = include_str!("test_nl.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("nl").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Vincent van Gogh", "vincent van gogh", 0.0111),
            ("Gogh Museum", "gogh museum", 0.0125),
            ("Gogh", "gogh", 0.0150),
            ("Museum", "museum", 0.0438),
            ("brieven", "brieven", 0.0635),
            ("Vincent", "vincent", 0.0643),
            ("Goghs schilderijen", "goghs schilderijen", 0.1009),
            ("Gogh verging", "gogh verging", 0.1215),
            ("Goghs", "goghs", 0.1651),
            ("schrijven", "schrijven", 0.1704),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn finnish_sample_defaults() {
        let text = include_str!("test_fi.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("fi").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Mobile Networks", "mobile networks", 0.0043),
            ("Nokia tekee muutoksia", "nokia tekee muutoksia", 0.0061),
            ("tekee muutoksia organisaatioonsa", "tekee muutoksia organisaatioonsa", 0.0065),
            ("johtokuntaansa vauhdittaakseen yhtiön", "johtokuntaansa vauhdittaakseen yhtiön", 0.0088),
            ("vauhdittaakseen yhtiön strategian", "vauhdittaakseen yhtiön strategian", 0.0088),
            ("yhtiön strategian toteuttamista", "yhtiön strategian toteuttamista", 0.0092),
            ("Networks", "networks", 0.0102),
            ("Networks and Applications", "networks and applications", 0.0113),
            ("strategian toteuttamista Nokia", "strategian toteuttamista nokia", 0.0127),
            ("siirtyy Mobile Networks", "siirtyy mobile networks", 0.0130),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn italian_sample_defaults() {
        let text = include_str!("test_it.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("it").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(5));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Champions League", "champions league", 0.0390),
            ("Quarti", "quarti", 0.0520),
            ("Atlético Madrid", "atlético madrid", 0.0592),
            ("Ottavi di finale", "ottavi di finale", 0.0646),
            ("Real Madrid", "real madrid", 0.0701),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn french_sample_defaults() {
        let text = include_str!("test_fr.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("fr").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("dégrade en France", "dégrade en france", 0.0254),
            ("jusque-là uniquement associée", "jusque-là uniquement associée", 0.0504),
            ("sondage Ifop réalisé", "sondage ifop réalisé", 0.0554),
            ("religion se dégrade", "religion se dégrade", 0.091),
            ("France", "france", 0.0941),
            ("l'extrême droite", "l'extrême droite", 0.0997),
            ("sondage Ifop", "sondage ifop", 0.101),
            ("Islam", "islam", 0.1021),
            ("musulmane en France", "musulmane en france", 0.1078),
            ("Allemagne", "allemagne", 0.1086),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    #[ignore = "Crashes due to failed unwrap"]
    fn portuguese_sport_sample_defaults() {
        let text = include_str!("test_pt_1.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("pt").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("seleção brasileira treinará", "seleção brasileira treinará", 0.0072),
            ("seleção brasileira", "seleção brasileira", 0.0100),
            ("Seleção Brasileira visando", "seleção brasileira visando", 0.0192),
            ("Seleção Brasileira encara", "seleção brasileira encara", 0.0344),
            ("brasileira treinará", "brasileira treinará", 0.0373),
            ("Renato Augusto", "renato augusto", 0.0376),
            ("Copa da Rússia", "copa da rússia", 0.0407),
            ("seleção", "seleção", 0.0454),
            ("brasileira", "brasileira", 0.0528),
        ];

        assert_eq!(actual, expected);
    }

    #[test]
    fn portuguese_tourism_sample_defaults() {
        let text = include_str!("test_pt_2.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("pt").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Alvor", "alvor", 0.0165),
            ("Rio Alvor", "rio alvor", 0.0336),
            ("Ria de Alvor", "ria de alvor", 0.0488),
            ("encantadora vila", "encantadora vila", 0.0575),
            ("Algarve", "algarve", 0.0774),
            ("impressionantes de Portugal", "impressionantes de portugal", 0.0844),
            ("estuário do Rio", "estuário do rio", 0.0907),
            ("vila", "vila", 0.1017),
            ("Ria", "ria", 0.1053),
            ("Oceano Atlântico", "oceano atlântico", 0.1357),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn spanish_sample_defaults() {
        let text = include_str!("test_es.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("es").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Guerra Civil Española", "guerra civil española", 0.0032),
            ("Guerra Civil", "guerra civil", 0.0130),
            ("Civil Española", "civil española", 0.0153),
            ("Partido Socialista Obrero", "partido socialista obrero", 0.0283),
            ("empezó la Guerra", "empezó la guerra", 0.0333),
            ("Socialista Obrero Español", "socialista obrero español", 0.0411),
            ("José Castillo", "josé castillo", 0.0426),
            ("Española", "española", 0.0566),
            ("José Antonio Primo", "josé antonio primo", 0.0589),
            ("José Calvo Sotelo", "josé calvo sotelo", 0.0596),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn polish_sample_defaults() {
        let text = include_str!("test_pl.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("pl").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("franka", "franka", 0.0328),
            ("Geerta Wildersa VVD", "geerta wildersa vvd", 0.0346),
            ("Geerta Wildersa", "geerta wildersa", 0.0399),
            ("kurs franka", "kurs franka", 0.0486),
            ("partii Geerta Wildersa", "partii geerta wildersa", 0.0675),
            ("proc", "proc", 0.0692),
            ("mld", "mld", 0.0724),
            ("Narodowego Banku Szwajcarii", "narodowego banku szwajcarii", 0.0728),
            ("kurs franka poniżej", "kurs franka poniżej", 0.0758),
            ("Wildersa", "wildersa", 0.0765),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn turkish_sample_defaults() {
        let text = include_str!("test_tr.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("tr").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("OECD", "oecd", 0.0176),                               // LIAAD REFERENCE: 0.0178
            ("Tek Bakışta Eğitim", "tek bakışta eğitim", 0.0232),   // LIAAD REFERENCE: 0.0236
            ("eğitim", "eğitim", 0.0274),                           // LIAAD REFERENCE: 0.0278
            ("OECD eğitim endeksi", "oecd eğitim endeksi", 0.0313), // LIAAD REFERENCE: 0.0323
            ("OECD ortalamasının", "oecd ortalamasının", 0.0375),   // LIAAD REFERNENCE: 0.0383
            ("Kalkınma Örgütü'nün", "kalkınma örgütü'nün", 0.0449), // LIAAD REFERENCE: 0.045
            ("Tek Bakışta", "tek bakışta", 0.0449),                 // LIAAD REFERENCE: 0.045
            ("İşbirliği ve Kalkınma", "i̇şbirliği ve kalkınma", 0.0468),
            ("Türkiye'de", "türkiye'de", 0.0476), // LIAAD REFERENCE: 0.0480
            ("yüksek", "yüksek", 0.0509),         // LIAAD REFERENCE: 0.0513
        ];

        // REASONS FOR DISCREPANCY:
        // - Difference in tokenization.

        assert_eq!(actual, expected);
    }

    #[test]
    fn arabic_sample_defaults() {
        let text = include_str!("test_ar.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("ar").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("عبد السلام العجيلي", "عبد السلام العجيلي", 0.0105),
            ("اللغة العربية الأربعاء", "اللغة العربية الأربعاء", 0.0139),
            ("عبد النبي اصطيف", "عبد النبي اصطيف", 0.0142),
            ("العجيلي في مرآة", "العجيلي في مرآة", 0.0177),
            ("مرآة النقد المقارن", "مرآة النقد المقارن", 0.0183), // LIAAD REFERENCE: 0.018
            ("السلام العجيلي", "السلام العجيلي", 0.0198),
            ("اللغة العربية", "اللغة العربية", 0.0207),
            ("مرآة النقد", "مرآة النقد", 0.0255), // LIAAD REFERENCE: 0.025
            ("اللغة العربية بدمشق", "اللغة العربية بدمشق", 0.0261),
            ("مجمع اللغة العربية", "مجمع اللغة العربية", 0.0281),
        ];

        assert_eq!(actual, expected);
    }

    #[test]
    fn dataset_text_1_defaults() {
        let text = include_str!("test_data_1.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("pt").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Médio Oriente continua", "médio oriente continua", 0.0008),
            ("Médio Oriente", "médio oriente", 0.0045),
            ("Oriente continua", "oriente continua", 0.0117),
            ("registar-se violentos confrontos", "registar-se violentos confrontos", 0.0178),
            ("Faixa de Gaza", "faixa de gaza", 0.0268),
            ("fogo hoje voltaram", "fogo hoje voltaram", 0.0311),
            ("voltaram a registar-se", "voltaram a registar-se", 0.0311),
            ("registar-se violentos", "registar-se violentos", 0.0311),
            ("Exército israelita", "exército israelita", 0.0368),
            ("Exército israelita voltou", "exército israelita voltou", 0.0639),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn dataset_text_2_defaults() {
        let text = include_str!("test_data_2.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(5));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("highly radioactive water", "highly radioactive water", 0.0006),
            ("crippled nuclear plant", "crippled nuclear plant", 0.0006),
            ("ocean Japan official", "ocean japan official", 0.0031),
            ("Japan official", "japan official", 0.0046),
            ("official says highly", "official says highly", 0.0050),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn dataset_text_3_defaults() {
        let text = include_str!("test_data_3.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(5));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Global Crossing", "global crossing", 0.0034),
            ("Hutchison Telecommunications", "hutchison telecommunications", 0.0053),
            ("Telecommunications and Singapore", "telecommunications and singapore", 0.0072),
            ("Singapore Technologies", "singapore technologies", 0.0072),
            ("Technologies take control", "technologies take control", 0.0157),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn dataset_text_4_defaults() {
        let text = include_str!("test_data_4.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("annual revenues increasing", "annual revenues increasing", 0.0018),
            ("retail inventory management", "retail inventory management", 0.0023),
            ("Dollar General", "dollar general", 0.0034),
            ("inventory management", "inventory management", 0.0112),
            ("perpetual progress", "perpetual progress", 0.0133),
            ("revenues increasing", "revenues increasing", 0.0133),
            ("fast track", "fast track", 0.0133),
            ("road to perpetual", "road to perpetual", 0.0159),
            ("annual revenues", "annual revenues", 0.0168),
            ("stores opened", "stores opened", 0.0168),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn dataset_text_5_defaults() {
        let text = include_str!("test_data_5.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Handoff Trigger Table", "handoff trigger table", 0.0006), // LIAAD REFERENCE: 0.0007
            ("Handoff", "handoff", 0.0010),
            ("WLAN Networks ABSTRACT", "wlan networks abstract", 0.0019),
            ("Vertical handoff", "vertical handoff", 0.0020),
            ("Handoff Trigger", "handoff trigger", 0.0021),
            ("proactive handoff scheme", "proactive handoff scheme", 0.0021),
            ("HTT Method Figure", "htt method figure", 0.0022),
            ("WLAN", "wlan", 0.0023),
            ("ABSTRACT Vertical handoff", "abstract vertical handoff", 0.0030),
            ("traditional handoff scheme", "traditional handoff scheme", 0.0033),
        ];

        // REASONS FOR DISCREPANCY:
        // - Difference in sentence splitting.

        assert_eq!(actual, expected);
    }

    #[test]
    fn dataset_text_6_defaults() {
        let text = include_str!("test_data_6.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("MRSA", "mrsa", 0.0047),
            ("TSN Database", "tsn database", 0.0107),
            ("methicillin-resistant Staphylococcus aureus", "methicillin-resistant staphylococcus aureus", 0.0116),
            ("rates of MRSA", "rates of mrsa", 0.0145),
            ("Staphylococcus aureus", "staphylococcus aureus", 0.0167),
            ("methicillin-resistant Staphylococcus", "methicillin-resistant staphylococcus", 0.0177),
            ("prevalence of MRSA", "prevalence of mrsa", 0.0201),
            ("MRSA infections", "mrsa infections", 0.0218),
            ("MRSA infections detected", "mrsa infections detected", 0.0223),
            ("TSN", "tsn", 0.0250),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }

    #[test]
    fn dataset_text_7_defaults() {
        let text = include_str!("test_data_7.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected = [
            ("Environment Design Level", "environment design level", 0.0008),
            ("Jerusalem Jerusalem", "jerusalem jerusalem", 0.0009),
            ("Dynamics Based Control", "dynamics based control", 0.0011),
            ("system dynamics", "system dynamics", 0.0017),
            ("DBC", "dbc", 0.0019),
            ("target system dynamics", "target system dynamics", 0.0019),
            ("target dynamics", "target dynamics", 0.0023),
            ("Science Bar Ilan", "science bar ilan", 0.0025),
            ("EMT", "emt", 0.0026),
            ("Dynamics", "dynamics", 0.0026),
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }
}
