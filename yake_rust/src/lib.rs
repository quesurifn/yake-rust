#![allow(clippy::len_zero)]
#![allow(clippy::type_complexity)]

use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::ops::Deref;

use indexmap::{IndexMap, IndexSet};
use preprocessor::{split_into_sentences, split_into_words};
#[cfg(feature = "serde")]
use serde;
use stats::{mean, median, stddev};

use crate::levenshtein::levenshtein_ratio;
pub use crate::stopwords::StopWords;

mod levenshtein;
mod preprocessor;
mod stopwords;

/// String from the original text
type RawString = String;

/// Lowercased string
type LString = String;

/// Lowercased string without punctuation symbols in single form
type UTerm = String;

type Sentences = Vec<Sentence>;
/// Key is `stems.join(" ")`
type Candidates<'s> = IndexMap<&'s [LString], PreCandidate<'s>>;
type Features<'s> = HashMap<&'s LString, YakeCandidate>;
type Words<'s> = HashMap<&'s UTerm, Vec<Occurrence<'s>>>;
type Contexts<'s> = HashMap<&'s UTerm, (Vec<&'s UTerm>, Vec<&'s RawString>)>;

struct WeightedCandidates<'s> {
    final_weights: IndexMap<&'s [LString], f64>,
    _surface_to_lexical: HashMap<&'s [LString], &'s [LString]>,
    _contexts: Contexts<'s>,
    raw_lookup: HashMap<&'s [LString], &'s [RawString]>,
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
    fn is_acronym(&self) -> bool {
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
struct YakeCandidate {
    is_stopword: bool,
    /// Term frequency. The total number of occurrences in the text.
    tf: f64,
    /// The number of times this candidate term is marked as an acronym (=all uppercase).
    tf_a: f64,
    /// The number of occurrences of this candidate term starting with an uppercase letter.
    tf_u: f64,
    /// Term casing heuristic.
    casing: f64,
    /// Term position heuristic
    position: f64,
    /// Normalized term frequency heuristic
    frequency: f64,
    /// Left dispersion
    dl: f64,
    /// Right dispersion
    dr: f64,
    /// Term relatedness to context
    relatedness: f64,
    /// Term's different sentences heuristic
    sentences: f64,
    /// Importance score. The less, the better
    weight: f64,
}

#[derive(PartialEq, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResultItem {
    pub raw: String,
    pub keyword: LString,
    pub score: f64,
}

#[derive(Debug, Clone)]
struct Sentence {
    pub words: Vec<RawString>,
    pub is_punctuation: Vec<bool>,
    pub lc_words: Vec<LString>,
    pub uq_terms: Vec<UTerm>,
    pub length: usize,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct PreCandidate<'s> {
    pub surfaces: Vec<&'s [String]>,
    pub lc_surfaces: Vec<&'s [LString]>,
    pub uq_terms: Vec<&'s [UTerm]>,
    pub offsets: Vec<usize>,
    pub sentence_ids: Vec<usize>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        self.filter_candidates(&mut ngrams, 3, 2, 5, false);
        let weighted_candidates = self.candidate_weighting(features, context, ngrams);

        let mut results = weighted_candidates
            .final_weights
            .into_iter()
            .map(|(keyword, score)| {
                let raw = weighted_candidates.raw_lookup.get(&keyword).unwrap().join(" ");
                let keyword = keyword.join(" ");
                ResultItem { raw, keyword, score }
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
        let mut unique_term = word.to_single().to_lowercase();
        for punctuation_symbol in &self.config.punctuation {
            unique_term = unique_term.replace(*punctuation_symbol, "");
        }
        unique_term
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
                let lc_words = words.iter().map(|w| w.to_lowercase()).collect::<Vec<LString>>();
                let uq_terms = lc_words.iter().map(|w| self.get_unique_term(w)).collect();
                let is_punctuation = words.iter().map(|w| self.word_is_punctuation(w)).collect();
                Sentence { length: words.len(), words, lc_words, uq_terms, is_punctuation }
            })
            .collect()
    }

    fn build_vocabulary<'s>(&self, sentences: &'s [Sentence]) -> Words<'s> {
        let mut words = Words::new();

        for (idx, sentence) in sentences.iter().enumerate() {
            let shift = sentences[0..idx].iter().map(|s| s.length).sum::<usize>();

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
        let mut contexts = Contexts::new();

        for sentence in sentences {
            let mut buffer: Vec<&UTerm> = Vec::new();

            for ((snt_word, snt_key), &is_punctuation) in
                sentence.words.iter().zip(&sentence.uq_terms).zip(&sentence.is_punctuation)
            {
                if is_punctuation {
                    buffer.clear();
                    continue;
                }

                let min_range = buffer.len().saturating_sub(self.config.window_size);
                let max_range = buffer.len();

                for &buf_word in &buffer[min_range..max_range] {
                    let entry_1 = contexts.entry(snt_key).or_default();
                    entry_1.0.push(buf_word);

                    let entry_2 = contexts.entry(buf_word).or_default();
                    entry_2.1.push(snt_word);
                }
                buffer.push(snt_key);
            }
        }

        contexts
    }

    /// Computes local statistic features that extract informative content within the text
    /// to calculate the importance of single terms.
    fn extract_features<'s>(&self, contexts: &Contexts, words: Words<'s>, sentences: &'s Sentences) -> Features<'s> {
        let tf = words.values().map(Vec::len);

        let words_nsw: HashMap<&UTerm, usize> = sentences
            .iter()
            .flat_map(|sentence| sentence.lc_words.iter().zip(&sentence.uq_terms).zip(&sentence.is_punctuation))
            .filter(|&((lc_word, _), is_punct)| !is_punct && !self.stop_words.contain(lc_word))
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
            .flat_map(|sentence| sentence.lc_words.iter().zip(&sentence.uq_terms).zip(&sentence.is_punctuation))
            .filter(|&(_, is_punct)| !is_punct)
            .map(|(pair, _)| pair)
            .collect();

        for (lc_word, u_term) in candidate_words {
            let occurrences = words.get(u_term).unwrap();

            let mut cand = YakeCandidate {
                is_stopword: self.stop_words.contain(lc_word),
                tf: occurrences.len() as f64,
                ..Default::default()
            };

            {
                // We consider the occurrence of acronyms through a heuristic, where all the letters of the word are capitals.
                cand.tf_a = occurrences.iter().filter(|&occ| occ.is_acronym()).count() as f64;

                // We give extra attention to any term beginning with a capital letter (excluding the beginning of sentences).
                let is_capital =
                    if self.config.strict_capital { Occurrence::is_capitalized } else { Occurrence::is_uppercased };

                cand.tf_u = occurrences.iter().filter(|&occ| is_capital(occ) && !occ.is_first_word()).count() as f64;

                // The casing aspect of a term is an important feature when considering the extraction
                // of keywords. The underlying rationale is that uppercase terms tend to be more
                // relevant than lowercase ones.
                cand.casing = cand.tf_a.max(cand.tf_u);

                // The more often the candidate term occurs with a capital letter, the more important
                // it is considered. This means that a candidate term that occurs with a capital letter
                // ten times within ten occurrences will be given a higher value (4.34) than a candidate
                // term that occurs with a capital letter five times within five occurrences (3.10).
                cand.casing /= 1.0 + cand.tf.ln(); // todo: no 1+ in the paper
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
                // scores to terms found in early sentences. todo: set affects median
                let sentence_ids: HashSet<_> = occurrences.iter().map(|o| o.idx).collect();
                // When the candidate term only appears in the first sentence, the median function
                // can return a value of 0. To guarantee position > 0, a constant 3 > e=2.71 is added.
                cand.position = 3.0 + median(sentence_ids.into_iter()).unwrap();
                // A double log is applied in order to smooth the difference between terms that occur
                // with a large median difference.
                cand.position = cand.position.ln().ln();
            }

            {
                // The higher the frequency of a candidate term, the greater its importance.
                cand.frequency = cand.tf;
                // To prevent a bias towards high frequency in long documents, we balance it.
                // The mean and the standard deviation is calculated from non-stopwords terms,
                // as stopwords usually have high frequencies.
                cand.frequency /= mean_tf + std_tf;
            }

            {
                if let Some((leftward, rightward)) = contexts.get(&u_term) {
                    let distinct: HashSet<&UTerm> = HashSet::from_iter(leftward.iter().map(Deref::deref));
                    cand.dl = if leftward.is_empty() { 0. } else { distinct.len() as f64 / leftward.len() as f64 };

                    let distinct: HashSet<&RawString> = HashSet::from_iter(rightward.iter().map(Deref::deref));
                    cand.dr = if rightward.is_empty() { 0. } else { distinct.len() as f64 / rightward.len() as f64 };
                }

                cand.relatedness = 1.0 + (cand.dr + cand.dl) * (cand.tf / max_tf);
            }

            {
                // Candidates which appear in many different sentences have a higher probability
                // of being important.
                let distinct = occurrences.iter().map(|o| o.idx).collect::<HashSet<_>>();
                cand.sentences = distinct.len() as f64 / sentences.len() as f64;
            }

            cand.weight = (cand.relatedness * cand.position)
                / (cand.casing + (cand.frequency / cand.relatedness) + (cand.sentences / cand.relatedness));

            features.insert(lc_word, cand);
        }

        features
    }

    fn candidate_weighting<'s>(
        &self,
        features: Features<'s>,
        contexts: Contexts<'s>,
        candidates: Candidates<'s>,
    ) -> WeightedCandidates<'s> {
        let mut final_weights: IndexMap<&'s [String], f64> = IndexMap::new();
        let mut _surface_to_lexical = HashMap::new();
        let mut raw_lookup = HashMap::new();

        for (_k, v) in candidates {
            for (&candidate, &u_terms) in v.lc_surfaces.iter().zip(&v.uq_terms) {
                let tokens = candidate;
                let mut prod_ = 1.0;
                let mut sum_ = 0.0;

                for (j, (token, term_stop)) in tokens.iter().zip(u_terms).enumerate() {
                    let Some(feat_cand) = features.get(token) else { continue };
                    if feat_cand.is_stopword {
                        let mut prob_t1 = 0.0;
                        let mut prob_t2 = 0.0;
                        if 1 < j {
                            let term_left = u_terms.get(j - 1).unwrap();
                            prob_t1 = contexts.get(&term_left).unwrap().1.iter().filter(|w| **w == term_stop).count()
                                as f64
                                / features.get(&term_left).unwrap().tf;
                        }
                        if j + 1 < tokens.len() {
                            let term_right = u_terms.get(j + 1).unwrap();
                            prob_t2 = contexts.get(&term_stop).unwrap().0.iter().filter(|w| **w == term_right).count()
                                as f64
                                / features.get(&term_right).unwrap().tf;
                        }

                        let prob = prob_t1 * prob_t2;
                        prod_ *= 1.0 + (1.0 - prob);
                        sum_ -= 1.0 - prob;
                    } else {
                        prod_ *= feat_cand.weight;
                        sum_ += feat_cand.weight;
                    }
                }
                if sum_ == -1.0 {
                    sum_ = 0.999999999;
                }

                let tf = v.surfaces.len() as f64;
                let weight = prod_ / (tf * (1.0 + sum_));

                final_weights.insert(candidate, weight);
                _surface_to_lexical.insert(candidate, *v.lc_surfaces.last().unwrap());
                raw_lookup.insert(candidate, v.surfaces[0]);
            }
        }

        WeightedCandidates { final_weights, _surface_to_lexical, _contexts: contexts, raw_lookup }
    }

    fn filter_candidates(
        &self,
        candidates: &mut Candidates,
        minimum_length: usize,
        minimum_word_size: usize,
        maximum_word_number: usize,
        only_alphanumeric_and_hyphen: bool, // could be a function
    ) {
        // fixme: filter right before inserting into the set to optimize
        candidates.retain(|_k, v| !{
            // get the words from the first occurring surface form
            let first_lc_surface = v.lc_surfaces[0];
            let last_lc_surface = v.lc_surfaces.last().unwrap();
            let lc_words: HashSet<&LString> = HashSet::from_iter(first_lc_surface);

            let has_float = || lc_words.iter().any(|w| w.parse::<f64>().is_ok());
            let has_stop_word = || self.stop_words.intersect_with(&lc_words);
            let is_punctuation = || lc_words.iter().any(|w| self.word_is_punctuation(w));
            let not_enough_symbols = || lc_words.iter().map(|w| w.len()).sum::<usize>() < minimum_length;
            let has_too_short_word = || lc_words.iter().map(|w| w.len()).min().unwrap_or(0) < minimum_word_size;
            let has_non_alphanumeric =
                || only_alphanumeric_and_hyphen && !lc_words.iter().all(word_is_alphanumeric_and_hyphen);

            // remove candidate if
            has_float()
                || has_stop_word()
                || is_punctuation()
                || not_enough_symbols()
                || has_too_short_word()
                || last_lc_surface.len() > maximum_word_number
                || has_non_alphanumeric()
                || first_lc_surface[0].len() < 3 // fixme: magic constant
                || first_lc_surface.last().unwrap().len() < 3
        });
    }

    fn ngram_selection<'s>(&self, n: usize, sentences: &'s Sentences) -> Candidates<'s> {
        let mut candidates: IndexMap<&'s [LString], PreCandidate<'_>> = Candidates::new();
        for (idx, sentence) in sentences.iter().enumerate() {
            let shift = sentences[0..idx].iter().map(|s| s.length).sum::<usize>();

            for j in 0..sentence.length {
                for k in j + 1..min(j + 1 + min(sentence.length, n), sentence.length + 1) {
                    if (j..k).is_empty() {
                        continue;
                    }

                    let words = &sentence.words[j..k];
                    let lc_words = &sentence.lc_words[j..k];

                    let candidate = candidates.entry(lc_words).or_default();
                    candidate.surfaces.push(words);
                    candidate.lc_surfaces.push(lc_words);
                    candidate.uq_terms.push(&sentence.uq_terms);
                    candidate.sentence_ids.push(idx);
                    candidate.offsets.push(j + shift);
                }
            }
        }
        candidates
    }

    fn word_is_punctuation(&self, word: impl AsRef<str>) -> bool {
        HashSet::from_iter(word.as_ref().chars()).is_subset(&self.config.punctuation)
    }
}

fn word_is_alphanumeric_and_hyphen(word: impl AsRef<str>) -> bool {
    word.as_ref().chars().all(|ch| ch.is_alphanumeric() || ch == '-')
}

trait PluralHelper {
    /// Omit the last `s` symbol in a string.
    ///
    /// How to use: `some_string.to_lowercase().to_single()`
    fn to_single(self) -> Self;
}

impl<'a> PluralHelper for &'a str {
    fn to_single(self) -> &'a str {
        if self.len() > 3 {
            self.trim_end_matches(&['s', 'S'][..])
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Results = Vec<ResultItem>;

    #[test]
    fn short() {
        let text = "this is a keyword";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(1));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected: Results = vec![ResultItem { raw: "keyword".into(), keyword: "keyword".into(), score: 0.1583 }];
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
        let expected: Results = vec![
            ResultItem { raw: "Machine".into(), keyword: "machine".into(), score: 0.1583 },
            ResultItem { raw: "learning".into(), keyword: "learning".into(), score: 0.1583 },
        ];
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
        let expected: Results = vec![
            ResultItem { raw: "Apple".into(), keyword: "apple".into(), score: 0.1448 },
            ResultItem { raw: "laptop".into(), keyword: "laptop".into(), score: 0.1583 },
        ];
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
        let expected: Results = vec![
            ResultItem { raw: "headphones".into(), keyword: "headphones".into(), score: 0.1141 },
            ResultItem { raw: "Saturday".into(), keyword: "saturday".into(), score: 0.2111 },
            ResultItem { raw: "Starting".into(), keyword: "starting".into(), score: 0.4096 },
        ];
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
        let expected: Results =
            vec![ResultItem { raw: "great deal".into(), keyword: "great deal".into(), score: 0.0257 }];
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
        let expected: Results = vec![
            ResultItem { raw: "smartwatch".into(), keyword: "smartwatch".into(), score: 0.2025 },
            ResultItem { raw: "phone".into(), keyword: "phone".into(), score: 0.2474 },
        ];
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
        let expected: Results = vec![
            ResultItem { raw: "smartwatch".into(), keyword: "smartwatch".into(), score: 0.2025 },
            ResultItem { raw: "phone".into(), keyword: "phone".into(), score: 0.4949 },
            ResultItem { raw: "phones".into(), keyword: "phones".into(), score: 0.4949 },
        ];
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
        let expected: Results =
            vec![ResultItem { raw: "high tech".into(), keyword: "high tech".into(), score: 0.0494 }];
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
        let expected: Results =
            vec![ResultItem { raw: "high-tech".into(), keyword: "high-tech".into(), score: 0.1583 }];
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
        let expected: Results = vec![
            ResultItem { raw: "weekly newsletter".into(), keyword: "weekly newsletter".into(), score: 0.0494 },
            ResultItem { raw: "newsletter".into(), keyword: "newsletter".into(), score: 0.1583 },
            ResultItem { raw: "weekly".into(), keyword: "weekly".into(), score: 0.2974 },
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
        let expected: Results = vec![
            ResultItem { raw: "weekly newsletter".into(), keyword: "weekly newsletter".into(), score: 0.0780 },
            ResultItem { raw: "newsletter".into(), keyword: "newsletter".into(), score: 0.2005 },
            ResultItem { raw: "weekly".into(), keyword: "weekly".into(), score: 0.3607 },
            ResultItem { raw: "great deals".into(), keyword: "great deals".into(), score: 0.4456 },
            ResultItem { raw: "high-tech drones".into(), keyword: "high-tech drones".into(), score: 0.4456 },
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
        let expected: Results = vec![
            ResultItem { raw: "weekly newsletter".into(), keyword: "weekly newsletter".into(), score: 0.0780 },
            ResultItem { raw: "newsletter".into(), keyword: "newsletter".into(), score: 0.2005 },
            ResultItem { raw: "weekly".into(), keyword: "weekly".into(), score: 0.3607 },
            ResultItem { raw: "great deals".into(), keyword: "great deals".into(), score: 0.4456 },
            ResultItem { raw: "high-tech drones".into(), keyword: "high-tech drones".into(), score: 0.4456 },
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
        let expected: Results = vec![
            ResultItem { raw: "Machine learning".into(), keyword: "machine learning".into(), score: 0.1346 },
            ResultItem { raw: "growing field".into(), keyword: "growing field".into(), score: 0.1672 },
            ResultItem { raw: "learning".into(), keyword: "learning".into(), score: 0.2265 },
            ResultItem { raw: "Machine".into(), keyword: "machine".into(), score: 0.2341 },
            ResultItem { raw: "growing".into(), keyword: "growing".into(), score: 0.2799 },
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
        let expected: Results = vec![
            ResultItem { raw: "yellow bananas".into(), keyword: "yellow bananas".into(), score: 0.1017 },
            ResultItem { raw: "day".into(), keyword: "day".into(), score: 0.1428 },
            ResultItem { raw: "bananas".into(), keyword: "bananas".into(), score: 0.1489 },
        ];

        // LIAAD REFERENCE:
        // - yellow bananas 0.0682
        // - buy 0.1428
        // - yellow 0.1428

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
        let expected: Results = vec![
            ResultItem { raw: "hundred yellow".into(), keyword: "hundred yellow".into(), score: 0.0446 },
            ResultItem { raw: "yellow bananas".into(), keyword: "yellow bananas".into(), score: 0.1017 },
            ResultItem { raw: "day".into(), keyword: "day".into(), score: 0.1428 },
        ];
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
        let expected: Results = vec![
            ResultItem { raw: "Google".into(), keyword: "google".into(), score: 0.0251 },
            ResultItem { raw: "Kaggle".into(), keyword: "kaggle".into(), score: 0.0273 },
            ResultItem { raw: "data".into(), keyword: "data".into(), score: 0.08 },
            ResultItem { raw: "science".into(), keyword: "science".into(), score: 0.0983 },
            ResultItem { raw: "platform".into(), keyword: "platform".into(), score: 0.124 },
            ResultItem { raw: "service".into(), keyword: "service".into(), score: 0.1316 },
            ResultItem { raw: "acquiring".into(), keyword: "acquiring".into(), score: 0.1511 },
            ResultItem { raw: "Goldbloom".into(), keyword: "goldbloom".into(), score: 0.1625 },
            ResultItem { raw: "machine".into(), keyword: "machine".into(), score: 0.1722 }, // LIAAD REFERENCE: 0.1672
            ResultItem { raw: "learning".into(), keyword: "learning".into(), score: 0.1722 }, // LIAAD REFERENCE: 0.1621 (so should be ranked higher)
        ];

        // REASONS FOR DISCREPANCY:
        // - "machine" and "learning" have different relatedness - in our code, they get one value; in LIAAD/yake, they
        //   both have different values, none of which matches ours.

        assert_eq!(actual, expected);
    }

    #[test]
    fn google_sample_defaults() {
        let text = include_str!("test_google.txt"); // LIAAD/yake sample text
        let stopwords = StopWords::predefined("en").unwrap();
        let mut actual = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));
        // leave only 4 digits
        actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let expected: Results = vec![
            ResultItem { raw: "Google".into(), keyword: "google".into(), score: 0.0251 },
            ResultItem { raw: "Kaggle".into(), keyword: "kaggle".into(), score: 0.0273 },
            ResultItem { raw: "CEO Anthony Goldbloom".into(), keyword: "ceo anthony goldbloom".into(), score: 0.0483 },
            ResultItem { raw: "data science".into(), keyword: "data science".into(), score: 0.055 },
            ResultItem {
                raw: "acquiring data science".into(),
                keyword: "acquiring data science".into(),
                score: 0.0603,
            },
            ResultItem { raw: "Google Cloud Platform".into(), keyword: "google cloud platform".into(), score: 0.0746 },
            ResultItem { raw: "data".into(), keyword: "data".into(), score: 0.08 },
            ResultItem { raw: "San Francisco".into(), keyword: "san francisco".into(), score: 0.0914 },
            ResultItem {
                raw: "Anthony Goldbloom declined".into(),
                keyword: "anthony goldbloom declined".into(),
                score: 0.0974,
            },
            ResultItem { raw: "science".into(), keyword: "science".into(), score: 0.0983 },
        ];
        // Results agree with reference implementation LIAAD/yake

        assert_eq!(actual, expected);
    }
}
