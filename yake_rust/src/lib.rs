#![allow(clippy::len_zero)]
#![allow(clippy::type_complexity)]

use crate::levenshtein::levenshtein_ratio;
use crate::preprocessor::PreprocessorCfg;
pub use crate::stopwords::StopWords;
use stats::{mean, median, stddev};
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::ops::Deref;

mod levenshtein;
mod preprocessor;
mod stopwords;

/// Lowercased string
type LString = String;

type Sentences = Vec<Sentence>;
/// Key is `stems.join(" ")`
type Candidates<'s> = HashMap<LString, PreCandidate<'s>>;
type Features = HashMap<LString, YakeCandidate>;
type Words<'s> = HashMap<LString, Vec<Occurrence<'s>>>;
type Contexts = HashMap<LString, (Vec<LString>, Vec<LString>)>;
type DedupeSubgram = HashMap<LString, bool>;

struct WeightedCandidates {
    final_weights: HashMap<LString, f64>,
    surface_to_lexical: HashMap<LString, String>,
    contexts: Contexts,
    raw_lookup: HashMap<LString, String>,
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

    fn is_capital(&self) -> bool {
        let mut chars = self.word.chars();
        chars.next().is_some_and(char::is_uppercase) // todo: what about the rest letters?
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
    weight: f64,
}

#[derive(PartialEq, Clone, Debug)]
pub struct ResultItem {
    pub raw: String,
    pub keyword: LString,
    pub score: f64,
}

#[derive(Debug, Clone)]
struct Sentence {
    pub words: Vec<String>,
    pub stems: Vec<LString>,
    pub length: usize,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct PreCandidate<'sentence> {
    pub surface_forms: Vec<&'sentence [String]>,
    pub lexical_form: &'sentence [LString],
    pub offsets: Vec<usize>,
    pub sentence_ids: Vec<usize>,
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            punctuation: r##"!"#$%&'()*+,-./:,<=>?@[\]^_`{|}~"##.chars().collect(),
            window_size: 1,
            deduplication_threshold: 0.9,
            ngrams: 3,
            remove_duplicates: true,
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

    pub fn get_n_best(&mut self, text: &str, n: Option<usize>) -> Vec<ResultItem> {
        let n = n.unwrap_or(10);

        let sentences = self.preprocess_text(text);
        let mut ngrams = self.ngram_selection(self.config.ngrams, &sentences);
        self.filter_candidates(&mut ngrams, None, None, None, None);

        let deduped_subgrams = self.candidate_selection(&mut ngrams);
        let vocabulary = self.build_vocabulary(&sentences);
        let context = self.build_context(&sentences);
        let features = self.extract_features(&context, vocabulary, &sentences);
        let weighted_candidates = self.candidate_weighting(features, context, ngrams, deduped_subgrams);

        let mut results = weighted_candidates
            .final_weights
            .into_iter()
            .map(|(keyword, score)| {
                let raw = weighted_candidates.raw_lookup.get(&keyword).unwrap().to_string();
                ResultItem { raw, keyword, score }
            })
            .collect::<Vec<_>>();

        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

        if self.config.remove_duplicates {
            self.remove_duplicates(results, n)
        } else {
            results.truncate(n);
            results
        }
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
        let cfg = PreprocessorCfg::default();
        preprocessor::split_into_sentences(text, &cfg)
            .into_iter()
            .map(|sentence| {
                let words = preprocessor::split_into_words(&sentence, &cfg);
                let stems = words.iter().map(|w| w.to_lowercase()).collect();
                Sentence { length: words.len(), words, stems }
            })
            .collect()
    }

    fn candidate_selection<'s>(&self, candidates: &mut Candidates<'s>) -> DedupeSubgram {
        let mut deduped = DedupeSubgram::new();

        candidates.retain(|_k, v| !{
            let first_surf_form = &v.surface_forms[0];

            if first_surf_form.len() > 1 {
                deduped.extend(first_surf_form.iter().map(|word| (word.to_lowercase(), true)));
            }

            let (fst, lst) = (&first_surf_form[0], first_surf_form.last().unwrap());

            // remove candidate if
            fst.len() < 3 || lst.len() < 3
        });

        deduped
    }

    fn build_vocabulary<'s>(&self, sentences: &'s [Sentence]) -> Words<'s> {
        let mut words = Words::new();

        for (idx, sentence) in sentences.iter().enumerate() {
            let shift = sentences[0..idx].iter().map(|s| s.length).sum::<usize>();

            for (w_idx, word) in sentence.words.iter().enumerate() {
                if !word.is_empty()
                    && word.chars().all(char::is_alphanumeric)
                    && HashSet::from_iter(word.chars()).intersection(&self.config.punctuation).next().is_none()
                {
                    let index = word.to_lowercase();
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
    fn build_context(&self, sentences: &Sentences) -> Contexts {
        let mut contexts = Contexts::new();

        for sentence in sentences {
            let mut buffer: Vec<LString> = Vec::new();
            let snt_words: Vec<LString> = sentence.words.iter().map(|w| w.to_lowercase()).collect();

            for snt_word in snt_words {
                if HashSet::from_iter(snt_word.chars()).is_subset(&self.config.punctuation) {
                    buffer.clear();
                    continue;
                }

                let min_range = buffer.len().saturating_sub(self.config.window_size);
                let max_range = buffer.len();

                for buf_word in &buffer[min_range..max_range] {
                    let entry_1 = contexts.entry(snt_word.clone()).or_default();
                    entry_1.0.push(buf_word.clone());

                    let entry_2 = contexts.entry(buf_word.clone()).or_default();
                    entry_2.1.push(snt_word.clone());
                }
                buffer.push(snt_word);
            }
        }

        contexts
    }

    /// Computes local statistic features that extract informative content within the text
    /// to calculate the importance of single terms.
    fn extract_features<'s>(&self, contexts: &Contexts, words: Words<'s>, sentences: &'s Sentences) -> Features {
        let tf = words.values().map(Vec::len);
        let tf_nsw = words
            .iter()
            // fixme: add len < 3 for stopwords
            .filter_map(|(k, v)| if !self.stop_words.contains(&k.to_owned()) { Some(v.len()) } else { None })
            .map(|x| x as f64)
            .collect::<Vec<_>>();

        let std_tf = stddev(tf_nsw.iter().copied());
        let mean_tf = mean(tf_nsw.iter().copied());
        let max_tf = tf.max().unwrap() as f64;

        let mut features = Features::new();

        for (key, occurrences) in words.into_iter() {
            let mut cand = YakeCandidate {
                is_stopword: self.stop_words.contains(&key) || key.len() < 3,
                tf: occurrences.len() as f64,
                ..Default::default()
            };

            {
                // We consider the occurrence of acronyms through a heuristic, where all the letters of the word are capitals.
                cand.tf_a = occurrences.iter().filter(|&occ| occ.is_acronym()).count() as f64;

                // We give extra attention to any term beginning with a capital letter (excluding the beginning of sentences).
                cand.tf_u = occurrences.iter().filter(|&occ| occ.is_capital() && !occ.is_first_word()).count() as f64;

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
                if let Some((leftward, rightward)) = contexts.get(&key) {
                    let distinct: HashSet<&str> = HashSet::from_iter(leftward.iter().map(Deref::deref));
                    cand.dl = if leftward.is_empty() { 0. } else { distinct.len() as f64 / leftward.len() as f64 };

                    let distinct: HashSet<&str> = HashSet::from_iter(rightward.iter().map(Deref::deref));
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

            features.insert(key, cand);
        }

        features
    }

    fn candidate_weighting(
        &self,
        features: Features,
        contexts: Contexts,
        candidates: Candidates,
        dedupe_subgram: DedupeSubgram,
    ) -> WeightedCandidates {
        let mut final_weights = HashMap::new();
        let mut surface_to_lexical = HashMap::new();
        let mut raw_lookup = HashMap::new();

        for (_k, v) in candidates {
            let lowercase_forms = v.surface_forms.iter().map(|w| w.join(" ").to_lowercase() as LString);
            for (idx, candidate) in lowercase_forms.enumerate() {
                let tokens = v.surface_forms[idx].iter().clone().map(|w| w.to_lowercase());
                let mut prod_ = 1.0;
                let mut sum_ = 0.0;

                // Dedup Subgram; Penalize subgrams
                if dedupe_subgram.contains_key(&candidate) {
                    prod_ += 5.0;
                }

                for (j, token) in tokens.clone().enumerate() {
                    let Some(feat_cand) = features.get(&token) else { continue };
                    if feat_cand.is_stopword {
                        let term_stop = token;
                        let mut prob_t1 = 0.0;
                        let mut prob_t2 = 0.0;
                        if 1 < j {
                            let term_left = tokens.clone().nth(j - 1).unwrap();
                            prob_t1 = contexts.get(&term_left).unwrap().1.iter().filter(|w| **w == term_stop).count()
                                as f64
                                / features.get(&term_left).unwrap().tf;
                        }
                        if j + 1 < tokens.len() {
                            let term_right = tokens.clone().nth(j + 1).unwrap();
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

                let tf = v.surface_forms.len() as f64;
                let weight = prod_ / (tf * (1.0 + sum_));

                final_weights.insert(candidate.clone(), weight);
                surface_to_lexical.insert(candidate.clone(), v.lexical_form.join(" "));
                raw_lookup.insert(candidate.clone(), v.surface_forms[0].join(" ").clone());
            }
        }

        WeightedCandidates { final_weights, surface_to_lexical, contexts, raw_lookup }
    }

    fn filter_candidates(
        &self,
        candidates: &mut Candidates,
        minimum_length: Option<usize>,
        minimum_word_size: Option<usize>,
        maximum_word_number: Option<usize>,
        only_alphanum: Option<bool>,
    ) {
        let minimum_length = minimum_length.unwrap_or(3);
        let minimum_word_size = minimum_word_size.unwrap_or(2);
        let maximum_word_number = maximum_word_number.unwrap_or(5);
        let only_alphanum = only_alphanum.unwrap_or(false); // fixme: replace with a function

        let in_char_set = |word: &str| word.chars().all(|ch| ch.is_alphanumeric() || ch == '-');
        let is_punctuation =
            |word: &str| HashSet::from_iter(word.chars()).intersection(&self.config.punctuation).next().is_some();

        // fixme: filter right before inserting into the set
        candidates.retain(|_k, v| !{
            // get the words from the first occurring surface form
            let first_surf_form = v.surface_forms[0];
            let words = HashSet::from_iter(first_surf_form.iter().map(|w| w.to_lowercase()));

            let has_float = || words.iter().any(|w| w.parse::<f64>().is_ok());
            let has_stop_word = || words.intersection(&self.stop_words).next().is_some();
            let has_punctuation = || words.iter().any(|w| is_punctuation(w));
            let not_enough_symbols = || words.iter().map(|w| w.len()).sum::<usize>() < minimum_length;
            let has_too_short_word = || words.iter().map(|w| w.len()).min().unwrap_or(0) < minimum_word_size;
            let has_non_alphanumeric = || only_alphanum && words.iter().any(|w| !in_char_set(w));

            // remove candidate if
            has_float()
                || has_stop_word()
                || has_punctuation()
                || not_enough_symbols()
                || has_too_short_word()
                || v.lexical_form.len() > maximum_word_number
                || has_non_alphanumeric()
        });
    }

    fn ngram_selection<'s>(&self, n: usize, sentences: &'s Sentences) -> Candidates<'s> {
        let mut candidates = Candidates::new();
        for (idx, sentence) in sentences.iter().enumerate() {
            let skip = min(n, sentence.length);
            let shift = sentences[0..idx].iter().map(|s| s.length).sum::<usize>();

            for j in 0..sentence.length {
                for k in j + 1..min(j + 1 + skip, sentence.length + 1) {
                    let words = &sentence.words[j..k];
                    let stems = &sentence.stems[j..k];
                    let sentence_id = idx;
                    let offset = j + shift;
                    let lexical_form = stems.join(" ");

                    let candidate = candidates.entry(lexical_form).or_default();
                    candidate.surface_forms.push(words);
                    candidate.sentence_ids.push(sentence_id);
                    candidate.offsets.push(offset);
                    candidate.lexical_form = stems;
                }
            }
        }
        candidates
    }
}

trait PluralHelper {
    /// Omit the last `s` symbol in a string.
    ///
    /// How to use: `some_string.to_lowercase().to_single()`
    fn to_single(self) -> String;
}

impl PluralHelper for String {
    fn to_single(mut self) -> String {
        if self.len() > 3 && self.ends_with('s') {
            self.truncate(self.len() - 1)
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Results = Vec<ResultItem>;

    #[test]
    fn keywords() {
        let text = include_str!("test_google.txt");
        let stopwords = StopWords::predefined("en").unwrap();
        let mut kwds = Yake::new(stopwords, Config::default()).get_n_best(text, Some(10));

        // leave only 4 digits
        kwds.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);

        let results: Results = vec![
            ResultItem { raw: "CEO Anthony Goldbloom".into(), keyword: "ceo anthony goldbloom".into(), score: 0.0296 },
            ResultItem { raw: "San Francisco".into(), keyword: "san francisco".into(), score: 0.0484 },
            ResultItem {
                raw: "Anthony Goldbloom declined".into(),
                keyword: "anthony goldbloom declined".into(),
                score: 0.0605,
            },
            ResultItem { raw: "Google Cloud Platform".into(), keyword: "google cloud platform".into(), score: 0.0614 },
            ResultItem { raw: "founder CEO Anthony".into(), keyword: "founder ceo anthony".into(), score: 0.0672 },
            ResultItem { raw: "hosts data science".into(), keyword: "hosts data science".into(), score: 0.0806 },
            ResultItem { raw: "acquiring Kaggle".into(), keyword: "acquiring kaggle".into(), score: 0.0855 },
            ResultItem { raw: "CEO Anthony".into(), keyword: "ceo anthony".into(), score: 0.0885 },
            ResultItem { raw: "Anthony Goldbloom".into(), keyword: "anthony goldbloom".into(), score: 0.0905 },
            ResultItem {
                raw: "machine learning competitions".into(),
                keyword: "machine learning competitions".into(),
                score: 0.0953,
            },
        ];

        assert_eq!(kwds, results);
    }

    #[test]
    fn short() {
        let text = "this is a keyword";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut kwds = Yake::new(stopwords, Config::default()).get_n_best(text, Some(1));
        // leave only 4 digits
        kwds.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let results: Results = vec![ResultItem { raw: "keyword".into(), keyword: "keyword".into(), score: 0.1583 }];

        assert_eq!(kwds, results);
    }

    #[test]
    fn medium() {
        let text = "Do you like headphones? \
        Starting this Saturday, we will be kicking off a huge sale of headphones! \
        If you need headphones, we've got you coverered!";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut kwds = Yake::new(stopwords, Config { ngrams: 1, ..Default::default() }).get_n_best(text, Some(3));
        // leave only 4 digits
        kwds.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let results: Results = vec![
            ResultItem { raw: "headphones".into(), keyword: "headphones".into(), score: 0.1141 },
            ResultItem { raw: "Saturday".into(), keyword: "saturday".into(), score: 0.2111 },
            ResultItem { raw: "Starting".into(), keyword: "starting".into(), score: 0.4096 },
        ];

        assert_eq!(kwds, results);
    }

    #[test]
    fn medium_two() {
        let text = "Do you need an Apple laptop?";
        let stopwords = StopWords::predefined("en").unwrap();
        let mut kwds = Yake::new(stopwords, Config { ngrams: 1, ..Default::default() }).get_n_best(text, Some(2));
        // leave only 4 digits
        kwds.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
        let results: Results = vec![
            ResultItem { raw: "Apple".into(), keyword: "apple".into(), score: 0.1448 },
            ResultItem { raw: "laptop".into(), keyword: "laptop".into(), score: 0.1583 },
        ];

        assert_eq!(kwds, results);
    }
}
