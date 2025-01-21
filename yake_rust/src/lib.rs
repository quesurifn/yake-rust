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
                    // todo: optimize: if some checks have failed, we may skip ngrams, by j += k
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
mod tests;
