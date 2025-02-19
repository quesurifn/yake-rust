#![cfg_attr(not(doctest), doc = include_str!("../README.md"))]
#![allow(clippy::len_zero)]
#![allow(clippy::type_complexity)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::private_intra_doc_links)]
#![deny(unused_imports)]
#![warn(missing_docs)]
#![allow(clippy::needless_doctest_main)]

use std::collections::VecDeque;

use hashbrown::hash_set::Entry;
use hashbrown::{HashMap, HashSet};
use indexmap::IndexMap;
use itertools::Itertools;
use plural_helper::PluralHelper;
use preprocessor::{split_into_sentences, split_into_words};
use stats::{median, OnlineStats};

use crate::context::Contexts;
pub use crate::result_item::*;
pub use crate::stopwords::StopWords;
use crate::tag::*;

mod context;
mod counter;
mod plural_helper;
mod preprocessor;
mod result_item;
mod stopwords;
mod tag;

#[cfg(test)]
mod tests;

/// String from the original text
type RawString = String;

/// Lowercased string
type LTerm = String;

/// Lowercased string without punctuation symbols in single form
type UTerm = String;

type Sentences = Vec<Sentence>;
type Candidates<'s> = IndexMap<&'s [LTerm], Candidate<'s>>;
type Features<'s> = HashMap<&'s UTerm, TermScore>;
type Words<'s> = HashMap<&'s UTerm, Vec<Occurrence<'s>>>;

#[derive(PartialEq, Eq, Hash, Debug)]
struct Occurrence<'sentence> {
    /// Index (0..) of sentence where the term occur
    pub sentence_idx: usize,
    /// The word itself
    pub word: &'sentence RawString,
    pub tag: Tag,
}

#[derive(Debug, Default)]
struct TermScore {
    /// Term frequency. The total number of occurrences in the text.
    tf: f64,
    /// Importance score. The less, the better
    score: f64,
}

#[derive(Debug, Default)]
struct TermStats {
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

impl From<TermStats> for TermScore {
    fn from(stats: TermStats) -> Self {
        Self { tf: stats.tf, score: stats.score }
    }
}

#[derive(Debug, Clone)]
struct Sentence {
    pub words: Vec<RawString>,
    pub lc_terms: Vec<LTerm>,
    pub uq_terms: Vec<UTerm>,
    pub tags: Vec<Tag>,
}

/// N-gram, a sequence of N terms.
#[derive(Debug, Default, Clone)]
struct Candidate<'s> {
    pub occurrences: usize,
    pub raw: &'s [RawString],
    pub lc_terms: &'s [LTerm],
    pub uq_terms: &'s [UTerm],
    pub score: f64,
}

/// Fine-tunes keyword extraction.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// How many words a key phrase may contain.
    ///
    /// _n-gram_ is a contiguous sequence of _n_ words occurring in the text.
    pub ngrams: usize,
    /// List of punctuation symbols.
    ///
    /// They are known as _exclude chars_ in the [original implementation](https://github.com/LIAAD/yake).
    pub punctuation: std::collections::HashSet<char>,

    /// The number of tokens both preceding and following a term to calculate _term dispersion_ metric.
    pub window_size: usize,
    /// When `true`, calculate _term casing_ metric by counting capitalized terms _without_
    /// intermediate uppercase letters. Thus, `Paypal` is counted while `PayPal` is not.
    ///
    /// The [original implementation](https://github.com/LIAAD/yake) sticks with `true`.
    pub strict_capital: bool,

    /// When `true`, key phrases are allowed to have only alphanumeric characters and hyphen.
    pub only_alphanumeric_and_hyphen: bool,
    /// Key phrases can't be too short, less than `minimum_chars` in total.
    pub minimum_chars: usize,

    /// When `true`, similar key phrases are deduplicated.
    ///
    /// Key phrases are considered similar if their Levenshtein distance is greater than
    /// [`deduplication_threshold`](Config::deduplication_threshold).
    pub remove_duplicates: bool,
    /// A threshold in range 0..1. Equal strings have the distance equal to 1.
    ///
    /// Effective only when [`remove_duplicates`](Config::remove_duplicates) is `true`.
    pub deduplication_threshold: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            punctuation: r##"!"#$%&'()*+,-./:,<=>?@[\]^_`{|}~"##.chars().collect(),
            deduplication_threshold: 0.9,
            ngrams: 3,
            remove_duplicates: true,
            window_size: 1,
            strict_capital: true,
            only_alphanumeric_and_hyphen: false,
            minimum_chars: 3,
        }
    }
}

/// Extract the top N most important key phrases from the text.
///
/// If you need all the keywords, pass [`usize::MAX`].
pub fn get_n_best(n: usize, text: &str, stop_words: &StopWords, config: &Config) -> Vec<ResultItem> {
    Yake::new(stop_words.clone(), config.clone()).get_n_best(text, n)
}

#[derive(Debug, Clone)]
struct Yake {
    config: Config,
    stop_words: StopWords,
}

impl Yake {
    pub fn new(stop_words: StopWords, config: Config) -> Yake {
        Self { config, stop_words }
    }

    fn get_n_best(&self, text: &str, n: usize) -> Vec<ResultItem> {
        let sentences = self.preprocess_text(text);

        let (context, vocabulary) = self.build_context_and_vocabulary(&sentences);
        let features = self.extract_features(&context, vocabulary, &sentences);

        let mut ngrams: Candidates = self.ngram_selection(self.config.ngrams, &sentences);
        self.candidate_weighting(features, &context, &mut ngrams);

        let mut results: Vec<ResultItem> = ngrams.into_values().map(Into::into).collect();
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

        if self.config.remove_duplicates {
            remove_duplicates(self.config.deduplication_threshold, results, n)
        } else {
            results.truncate(n);
            results
        }
    }

    fn get_unique_term(&self, word: &str) -> UTerm {
        word.to_single().to_lowercase()
    }

    fn is_stopword(&self, lc_term: &LTerm) -> bool {
        self.stop_words.contains(lc_term)
            || self.stop_words.contains(lc_term.to_single())
            // having less than 3 non-punctuation symbols is typical for stop words
            || lc_term.to_single().chars().filter(|ch| !self.config.punctuation.contains(ch)).count() < 3
    }

    fn preprocess_text(&self, text: &str) -> Sentences {
        split_into_sentences(text)
            .into_iter()
            .map(|sentence| {
                let words = split_into_words(&sentence);
                let lc_terms = words.iter().map(|w| w.to_lowercase()).collect::<Vec<LTerm>>();
                let uq_terms = lc_terms.iter().map(|w| self.get_unique_term(w)).collect();
                let tags = words.iter().enumerate().map(|(w_idx, w)| Tag::from(w, w_idx == 0, &self.config)).collect();
                Sentence { words, lc_terms, uq_terms, tags }
            })
            .collect()
    }

    fn build_context_and_vocabulary<'s>(&self, sentences: &'s [Sentence]) -> (Contexts<'s>, Words<'s>) {
        let mut ctx = Contexts::default();
        let mut words = Words::new();

        for (idx, sentence) in sentences.iter().enumerate() {
            let mut window: VecDeque<(&UTerm, Tag)> = VecDeque::with_capacity(self.config.window_size + 1);

            for ((word, term), &tag) in sentence.words.iter().zip(&sentence.uq_terms).zip(&sentence.tags) {
                if tag == Tag::Punctuation {
                    window.clear();
                    continue;
                }

                let occurrence = Occurrence { sentence_idx: idx, word, tag };
                words.entry(term).or_default().push(occurrence);

                // Do not store in contexts in any way if the word (not the unique term) is tagged "d" or "u"
                if tag != Tag::Digit && tag != Tag::Unparsable {
                    for &(left_uterm, left_tag) in window.iter() {
                        if left_tag == Tag::Digit || left_tag == Tag::Unparsable {
                            continue;
                        }

                        ctx.track(left_uterm, term);
                    }
                }

                if window.len() == self.config.window_size {
                    window.pop_front();
                }
                window.push_back((term, tag));
            }
        }

        (ctx, words)
    }

    /// Computes local statistic features that extract informative content within the text
    /// to calculate the importance of single terms.
    fn extract_features<'s>(&self, ctx: &Contexts, words: Words<'s>, sentences: &'s Sentences) -> Features<'s> {
        let candidate_words: HashMap<_, _> = sentences
            .iter()
            .flat_map(|sentence| sentence.lc_terms.iter().zip(&sentence.uq_terms).zip(&sentence.tags))
            .filter(|&(_, &tag)| tag != Tag::Punctuation)
            .map(|p| p.0)
            .collect();

        let non_stop_words: HashMap<&UTerm, usize> = candidate_words
            .iter()
            .filter(|&(lc, _)| !self.is_stopword(lc))
            .map(|(_, &uq)| {
                let occurrences = words.get(uq).unwrap().len();
                (uq, occurrences)
            })
            .collect();

        let (nsw_tf_std, nsw_tf_mean) = {
            let tfs: OnlineStats = non_stop_words.values().map(|&freq| freq as f64).collect();
            (tfs.stddev(), tfs.mean())
        };

        let max_tf = words.values().map(Vec::len).max().unwrap_or(0) as f64;

        let mut features = Features::new();

        for (_, u_term) in candidate_words {
            let occurrences = words.get(u_term).unwrap();
            let mut stats = TermStats { tf: occurrences.len() as f64, ..Default::default() };

            {
                // We consider the occurrence of acronyms through a heuristic, where all the letters of the word are capitals.
                stats.tf_a = occurrences.iter().map(|occ| occ.tag).filter(|&tag| tag == Tag::Acronym).count() as f64;
                // We give extra attention to any term beginning with a capital letter (excluding the beginning of sentences).
                stats.tf_n = occurrences.iter().map(|occ| occ.tag).filter(|&tag| tag == Tag::Uppercase).count() as f64;

                // The casing aspect of a term is an important feature when considering the extraction
                // of keywords. The underlying rationale is that uppercase terms tend to be more
                // relevant than lowercase ones.
                stats.casing = stats.tf_a.max(stats.tf_n);

                // The more often the candidate term occurs with a capital letter, the more important
                // it is considered. This means that a candidate term that occurs with a capital letter
                // ten times within ten occurrences will be given a higher value (4.34) than a candidate
                // term that occurs with a capital letter five times within five occurrences (3.10).
                stats.casing /= 1.0 + stats.tf.ln();
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
                // scores to terms found in early sentences.
                let sentence_ids = occurrences.iter().map(|o| o.sentence_idx).dedup();
                // When the candidate term only appears in the first sentence, the median function
                // can return a value of 0. To guarantee position > 0, a constant 3 > e=2.71 is added.
                stats.position = 3.0 + median(sentence_ids).unwrap();
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
                stats.frequency /= nsw_tf_mean + nsw_tf_std;
            }

            {
                let (dl, dr) = ctx.dispersion_of(u_term);
                stats.relatedness = 1.0 + (dr + dl) * (stats.tf / max_tf);
            }

            {
                // Candidates which appear in many different sentences have a higher probability
                // of being important.
                let distinct = occurrences.iter().map(|o| o.sentence_idx).dedup().count();
                stats.sentences = distinct as f64 / sentences.len() as f64;
            }

            stats.score = (stats.relatedness * stats.position)
                / (stats.casing + (stats.frequency / stats.relatedness) + (stats.sentences / stats.relatedness));

            features.insert(u_term, stats.into());
        }

        features
    }

    fn candidate_weighting<'s>(&self, features: Features<'s>, ctx: &Contexts<'s>, candidates: &mut Candidates<'s>) {
        for (&lc_terms, candidate) in candidates.iter_mut() {
            let uq_terms = candidate.uq_terms;
            let mut prod_ = 1.0;
            let mut sum_ = 0.0;

            for (j, (lc, uq)) in lc_terms.iter().zip(uq_terms).enumerate() {
                if self.is_stopword(lc) {
                    let prob_prev = match uq_terms.get(j - 1) {
                        None => 0.0,
                        Some(prev_uq) => {
                            // #previous term occurring before this one / #previous term
                            ctx.cases_term_is_followed(prev_uq, uq) as f64 / features.get(&prev_uq).unwrap().tf
                        }
                    };

                    let prob_succ = match uq_terms.get(j + 1) {
                        None => 0.0,
                        Some(next_uq) => {
                            // #next term occurring after this one / #next term
                            ctx.cases_term_is_followed(uq, next_uq) as f64 / features.get(&next_uq).unwrap().tf
                        }
                    };

                    let prob = prob_prev * prob_succ;
                    prod_ *= 1.0 + (1.0 - prob);
                    sum_ -= 1.0 - prob;
                } else if let Some(stats) = features.get(uq) {
                    prod_ *= stats.score;
                    sum_ += stats.score;
                }
            }

            if sum_ == -1.0 {
                sum_ = 0.999999999;
            }

            let tf = candidate.occurrences as f64;
            candidate.score = prod_ / (tf * (1.0 + sum_));
        }
    }

    fn is_candidate(&self, lc_terms: &[LTerm], tags: &[Tag]) -> bool {
        let is_bad =
            // has a bad tag
            tags.iter().any(|tag| matches!(tag, Tag::Digit | Tag::Punctuation | Tag::Unparsable))
            // the last word is a stopword
            || self.is_stopword(lc_terms.last().unwrap())
            // not enough symbols in total
            || lc_terms.iter().map(|w| w.chars().count()).sum::<usize>() < self.config.minimum_chars
            // has non-alphanumeric characters
            || self.config.only_alphanumeric_and_hyphen && !lc_terms.iter().all(word_is_alphanumeric_and_hyphen);

        !is_bad
    }

    fn ngram_selection<'s>(&self, n: usize, sentences: &'s Sentences) -> Candidates<'s> {
        let mut candidates = Candidates::new();
        let mut ignored = HashSet::new();

        for sentence in sentences.iter() {
            let length = sentence.words.len();

            for j in 0..length {
                if self.is_stopword(&sentence.lc_terms[j]) {
                    continue; // all further n-grams starting with this word can't be candidates
                }

                for k in (j + 1..length + 1).take(n) {
                    let lc_terms = &sentence.lc_terms[j..k];

                    if let Entry::Vacant(e) = ignored.entry(lc_terms) {
                        if !self.is_candidate(lc_terms, &sentence.tags[j..k]) {
                            e.insert();
                        } else {
                            candidates
                                .entry(lc_terms)
                                .or_insert_with(|| Candidate {
                                    lc_terms,
                                    uq_terms: &sentence.uq_terms[j..k],
                                    raw: &sentence.words[j..k],
                                    ..Default::default()
                                })
                                .occurrences += 1;
                        }
                    };
                }
            }
        }

        candidates
    }
}

fn word_is_alphanumeric_and_hyphen(word: impl AsRef<str>) -> bool {
    word.as_ref().chars().all(|ch| ch.is_alphanumeric() || ch == '-')
}
