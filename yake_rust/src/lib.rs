#![allow(clippy::len_zero)]
#![allow(clippy::type_complexity)]

use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

use crate::levenshtein::levenshtein_ratio;
use crate::preprocessor::PreprocessorCfg;
use stats::{mean, median, stddev};

mod levenshtein;
mod preprocessor;

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
    pub shift_offset: usize,
    pub shift: usize,
    /// sentence index
    pub idx: usize,
    pub word: &'sentence String,
}

#[derive(Debug, Default)]
struct YakeCandidate {
    isstop: bool,
    tf: f64,
    tf_a: f64,
    tf_u: f64,
    casing: f64,
    position: f64,
    frequency: f64,
    wl: f64,
    wr: f64,
    pl: f64,
    pr: f64,
    different: f64,
    relatedness: f64,
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

impl Sentence {
    pub fn new(words: Vec<String>, stems: Vec<String>) -> Sentence {
        let length = words.len();
        Sentence { words, length, stems }
    }
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
    pub ngram: usize,
    /// List of punctuation symbols.
    pub punctuation: HashSet<String>,
    /// List of lowercased words to be filtered from the text.
    pub stop_words: HashSet<LString>,
    pub window_size: usize,
    pub remove_duplicates: bool,
    /// A threshold in range 0..1.
    pub dedupe_lim: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            stop_words: include_str!("stop_words.txt").lines().map(ToOwned::to_owned).collect(),
            punctuation: r##"!"#$%&'()*+,-./:,<=>?@[\]^_`{|}~"##.chars().map(|ch| ch.to_string()).collect(),
            window_size: 2,
            dedupe_lim: 0.8,
            ngram: 3,
            remove_duplicates: true,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Yake {
    config: Config,
}

impl Yake {
    pub fn new(config: Config) -> Yake {
        Self { config }
    }

    pub fn get_n_best(&mut self, text: String, n: Option<usize>) -> Vec<ResultItem> {
        let n = n.unwrap_or(10);

        let sentences = self.prepare_text(text);
        let mut ngrams = self.ngram_selection(self.config.ngram, &sentences);
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

            let is_duplicate =
                unique.iter().any(|it| levenshtein_ratio(&it.keyword, &res.keyword) > self.config.dedupe_lim);

            if !is_duplicate {
                unique.push(res);
            }
        }

        unique
    }

    fn prepare_text(&self, text: String) -> Sentences {
        let cfg = PreprocessorCfg::default();
        preprocessor::split_into_sentences(&text, &cfg)
            .into_iter()
            .map(|sentence| {
                let words = preprocessor::split_into_words(&sentence, &cfg);
                let stems = words.iter().map(|w| w.to_lowercase()).collect();
                Sentence::new(words, stems)
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
                if word.chars().all(char::is_alphanumeric)
                    && HashSet::from_iter(word.chars().map(|x| x.to_string()))
                        .intersection(&self.config.punctuation)
                        .next()
                        .is_none()
                {
                    let index = word.to_lowercase();
                    let occurrence = Occurrence { shift_offset: shift + w_idx, idx, word, shift };
                    words.entry(index).or_default().push(occurrence)
                }
            }
        }

        words
    }

    fn build_context(&self, sentences: &Sentences) -> Contexts {
        let mut contexts = Contexts::new();

        for sentence in sentences {
            let mut buffer: Vec<LString> = Vec::new();
            let snt_words: Vec<LString> = sentence.words.iter().map(|w| w.to_lowercase()).collect();

            for snt_word in snt_words {
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

    fn extract_features<'s>(&self, contexts: &Contexts, words: Words<'s>, sentences: &'s Sentences) -> Features {
        let tf = words.values().map(Vec::len);
        let tf_nsw = words
            .iter()
            .filter_map(|(k, v)| if !self.config.stop_words.contains(&k.to_owned()) { Some(v.len()) } else { None })
            .map(|x| x as f64)
            .collect::<Vec<_>>();

        let std_tf = stddev(tf_nsw.iter().copied());
        let mean_tf = mean(tf_nsw.iter().copied());
        let max_tf = tf.max().unwrap() as f64;

        let mut features = Features::new();

        for (key, word) in words.into_iter() {
            let mut cand = YakeCandidate {
                isstop: self.config.stop_words.contains(&key) || key.len() < 3,
                tf: word.len() as f64,
                tf_a: 0.,
                tf_u: 0.,
                ..Default::default()
            };

            for occurrence in word.iter() {
                if occurrence.word.chars().all(char::is_uppercase) && occurrence.word.len() > 1 {
                    cand.tf_a += 1.0;
                }
                if occurrence.word.chars().nth(0).unwrap_or(' ').is_uppercase()
                    && occurrence.shift != occurrence.shift_offset
                {
                    cand.tf_u += 1.0;
                }
            }

            cand.casing = cand.tf_a.max(cand.tf_u);
            cand.casing /= 1.0 + cand.tf.ln();

            let sentence_ids = word.iter().map(|o| o.idx).collect::<HashSet<usize>>();
            cand.position = (3.0 + median(sentence_ids.iter().copied()).unwrap()).ln();
            cand.position = cand.position.ln();

            cand.frequency = cand.tf;
            cand.frequency /= mean_tf + std_tf;

            cand.wl = 0.0;

            let ctx = contexts.get(&key).unwrap();
            let ctx_1_hash: HashSet<String> = HashSet::from_iter(ctx.clone().0);
            if ctx.0.len() > 0 {
                cand.wl = ctx_1_hash.len() as f64;
                cand.wl /= ctx.0.len() as f64;
            }
            cand.pl = ctx_1_hash.len() as f64 / max_tf;

            cand.wr = 0.0;
            let ctx_2_hash: HashSet<LString> = HashSet::from_iter(ctx.clone().1);
            if ctx.1.len() > 0 {
                cand.wr = ctx_2_hash.len() as f64;
                cand.wr /= ctx.1.len() as f64;
            }
            cand.pr = ctx_2_hash.len() as f64 / max_tf;

            cand.relatedness = 1.0;
            cand.relatedness += (cand.wr + cand.wl) * (cand.tf / max_tf);

            cand.different = sentence_ids.len() as f64;
            cand.different /= sentences.len() as f64;
            cand.weight = (cand.relatedness * cand.position)
                / (cand.casing + (cand.frequency / cand.relatedness) + (cand.different / cand.relatedness));

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
                    if feat_cand.isstop {
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

        // fixme: filter right before inserting into the set
        candidates.retain(|_k, v| !{
            // get the words from the first occurring surface form
            let first_surf_form = v.surface_forms[0];
            let words = HashSet::from_iter(first_surf_form.iter().map(|w| w.to_lowercase()));

            let has_float = || words.iter().any(|w| w.parse::<f64>().is_ok());
            let has_stop_word = || words.intersection(&self.config.stop_words).next().is_some();
            let has_punctuation = || words.intersection(&self.config.punctuation).next().is_some();
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

#[cfg(test)]
mod tests {
    use super::*;
    type Results = Vec<ResultItem>;

    #[test]
    fn keywords() {
        let text = r#"
        Google is acquiring data science community Kaggle. Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning
        competitions. Details about the transaction remain somewhat vague, but given that Google is hosting its Cloud
        Next conference in San Francisco this week, the official announcement could come as early as tomorrow.
        Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening.
        Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data scientists on its platform,
        was founded by Goldbloom  and Ben Hamner in 2010.
        The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank,
        it has managed to stay well ahead of them by focusing on its specific niche.
        The service is basically the de facto home for running data science and machine learning competitions.
        With Kaggle, Google is buying one of the largest and most active communities for data scientists - and with that,
        it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow
        and other projects). Kaggle has a bit of a history with Google, too, but that's pretty recent. Earlier this month,
        Google and Kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos.
        That competition had some deep integrations with the Google Cloud Platform, too. Our understanding is that Google
        will keep the service running - likely under its current name. While the acquisition is probably more about
        Kaggle's community than technology, Kaggle did build some interesting tools for hosting its competition
        and 'kernels', too. On Kaggle, kernels are basically the source code for analyzing data sets and developers can
        share this code on the platform (the company previously called them 'scripts').
        Like similar competition-centric sites, Kaggle also runs a job board, too. It's unclear what Google will do with
        that part of the service. According to Crunchbase, Kaggle raised $12.5 million (though PitchBook says it's $12.75)
        since its   launch in 2010. Investors in Kaggle include Index Ventures, SV Angel, Max Levchin, Naval Ravikant,
        Google chief economist Hal Varian, Khosla Ventures and Yuri Milner
        "#;

        let mut kwds = Yake::default().get_n_best(text.to_string(), Some(10));

        // leave only 4 digits
        kwds.iter_mut().for_each(|r| r.score = (r.score * 10_000.).trunc() / 10_000.);

        let results: Results = vec![
            ResultItem {
                raw: "CEO Anthony Goldbloom".to_owned(),
                keyword: "ceo anthony goldbloom".to_owned(),
                score: 0.0463,
            },
            ResultItem {
                raw: "Google Cloud Platform".to_owned(),
                keyword: "google cloud platform".to_owned(),
                score: 0.0566,
            },
            ResultItem { raw: "data science".to_owned(), keyword: "data science".to_owned(), score: 0.0599 },
            ResultItem {
                raw: "acquiring data science".to_owned(),
                keyword: "acquiring data science".to_owned(),
                score: 0.0735,
            },
            ResultItem {
                raw: "science community Kaggle".to_owned(),
                keyword: "science community kaggle".to_owned(),
                score: 0.0795,
            },
            ResultItem { raw: "San Francisco".to_owned(), keyword: "san francisco".to_owned(), score: 0.0909 },
            ResultItem { raw: "acquiring Kaggle".to_owned(), keyword: "acquiring kaggle".to_owned(), score: 0.0915 },
            ResultItem {
                raw: "Anthony Goldbloom declined".to_owned(),
                keyword: "anthony goldbloom declined".to_owned(),
                score: 0.0964,
            },
            ResultItem { raw: "Google Cloud".to_owned(), keyword: "google cloud".to_owned(), score: 0.0969 },
            ResultItem { raw: "Kaggle".to_owned(), keyword: "kaggle".to_owned(), score: 0.1167 },
        ];

        assert_eq!(kwds, results);
    }
}
