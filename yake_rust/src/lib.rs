use std::collections::{HashMap, HashSet};
use std::cmp::{min, max};
use std::iter::FromIterator;
use stats::{stddev, mean, median};

mod levenshtein;
mod preprocessor;
mod stopwords;

type Sentences = Vec<Sentence>;
type Candidates = HashMap<String, PreCandidate>;
type Features =  HashMap<String, YakeCandidate>;
type Words = HashMap<String, Vec<Occurrence>>;
type Contexts = HashMap<String, (Vec<String>, Vec<String>)>;


#[derive(PartialEq, Eq, Hash, Debug)]
struct Occurrence {
    pub shift_offset: usize,
    pub shift: usize,
    pub index: usize,
    pub word: String,
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

#[derive(Debug, Clone)]
pub struct ResultItem {
    raw: String,
    keyword: String,
    score: f64,
}
impl ResultItem {
    fn new(raw: String, keyword: String, score: f64) -> ResultItem {
        ResultItem {
            raw,
            keyword,
            score,
        }
    }
}

#[derive(Debug, Clone)]
struct Sentence {
    pub words: Vec<String>,
    pub stems: Vec<String>,
    pub length: usize,
}
impl Sentence {
    pub fn new(words: Vec<String>, stems:Option<Vec<String>>) -> Sentence {
        let length = words.len();
        let default_stems = stems.unwrap_or(Vec::<String>::new());
        Sentence {
            words,
            length,
            stems: default_stems,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
struct PreCandidate {
    pub surface_forms: Vec<Vec<String>>,
    pub lexical_form: Vec<String>,
    pub offsets: Vec<usize>,
    pub sentence_ids: Vec<usize>,
}

struct Config {
    pub ngram: usize,
    pub punctuation: HashSet<String>,
    pub stopwords: HashSet<String>,
    pub remove_duplicates: bool,

    window_size: usize,
    dedupe_lim: f64,
}

pub struct Yake {
    config : Config,
}
impl Yake {
    pub fn new(ngram: Option<usize>, remove_duplicaes: Option<bool>, punctuation: Option<HashSet<String>>, stopwords: Option<HashSet<String>>) -> Yake {
        let default_stopwords = stopwords.unwrap_or(stopwords::StopWords::new().words);
        let default_punctuation = punctuation.unwrap_or(HashSet::from_iter( vec!["!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ",", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~"].iter().map(|&s| s.to_string())));
        let default_ngram = ngram.unwrap_or(3);
        let default_remove_duplicates = remove_duplicaes.unwrap_or(true);
        Yake {
            config: Config {
                window_size: 2,
                ngram: default_ngram,
                dedupe_lim: 0.8,
                stopwords: default_stopwords,
                punctuation: default_punctuation,
                remove_duplicates: default_remove_duplicates,
            },
        }
    }

    pub fn get_n_best(&mut self, text: String, n: Option<usize>) -> Vec<ResultItem>{
        let default_n = n.unwrap_or(10);
        let sentences = self.build_text(text);
        let selected_ngrams = self.ngram_selection(self.config.ngram, sentences);
        let filtered_candidates = self.candidate_filtering(selected_ngrams.0, None, None, None, None, None);
        let selected_candidates = self.candidate_selection(filtered_candidates);
        let built_words = self.vocabulary_building(selected_ngrams.1);
        let built_contexts = self.context_building(built_words.0, built_words.1);
        let built_features = self.feature_extraction(built_contexts.0, built_contexts.1, built_contexts.2);
        let weighted_candidates = self.candidate_weighting(built_features.0, built_features.1, selected_candidates);

        let mut results_vec = weighted_candidates.0.clone().iter().map(|(k, v)| ResultItem::new(weighted_candidates.1.get(k).unwrap().to_string(),   k.to_string(), *v)).collect::<Vec<ResultItem>>();
        results_vec.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

        if self.config.remove_duplicates {
            let mut non_redundant_best = Vec::<ResultItem>::new();
            for candidate in results_vec {
                if self.is_redundant(candidate.clone().keyword, non_redundant_best.iter().map(|x| x.keyword.to_string()).collect::<Vec<String>>()) {
                    continue;
                }
                non_redundant_best.push(candidate);

                if non_redundant_best.len() >= default_n {
                    break;
                }
            }
            results_vec = non_redundant_best;
        }

        results_vec.iter().take(min(default_n, results_vec.len())).map(|x| ResultItem { raw: x.raw.to_owned(), keyword: x.keyword.to_owned(), score: x.score }).collect::<Vec<ResultItem>>()
    }
    

    fn build_text(&mut self, text: String) -> Sentences {
        let mut sentences = Vec::<Sentence>::new();
        let preprocessor = preprocessor::Preprocessor::new(text, None, None).split_into_sentences();
        for sentence in preprocessor {
            let words = preprocessor::Preprocessor::new(sentence.to_string(), None, None).split_into_words();
            let stems = words.iter().map(|w| w.to_lowercase()).collect::<Vec<String>>();
            let sentence = Sentence::new(words, Some(stems));
            sentences.push(sentence);
        }
        sentences
    }

    fn candidate_selection(&mut self, mut candidates: HashMap<String, PreCandidate>) -> HashMap<String, PreCandidate> {
        for (k, v) in candidates.clone() {
            if  self.config.stopwords.contains(&v.surface_forms[0][0].to_lowercase()) ||
                self.config.stopwords.contains(&v.surface_forms[0].last().unwrap().to_lowercase()) || 
                v.surface_forms[0][0].len() < 3 ||
                v.surface_forms[0].last().unwrap().len() < 3 
            {
                candidates.remove(&k);
            }
        }
        candidates
    }

    fn vocabulary_building(&mut self, sentences: Vec<Sentence>) -> (Words, Sentences) {
        let mut words = HashMap::<String, Vec<Occurrence>>::new();
        for (idx, sentence) in sentences.clone().iter().enumerate() {
            let shift = sentences[0..idx].iter().map(|s| s.length).sum::<usize>(); 

            for (w_idx, word) in sentence.words.iter().enumerate() {
                if self.is_alphanum(word.to_string(), None) && HashSet::from_iter(word.split("").map(|x| x.to_string() )).intersection(&self.config.punctuation).count() == 0 {
                    let index = word.to_lowercase();
                    let new_occurrence = Occurrence {
                        shift_offset: shift + w_idx,
                        index: idx,
                        word: word.to_string(),
                        shift
                    };
                    
                    let object = words.get_mut(&index);
                    if object != None {
                        object.unwrap().push(new_occurrence)
                    } else {
                        words.insert( index, vec![new_occurrence]);
                    }
                }
            }
        }

        (words, sentences)
    }

    fn context_building(&mut self, words: Words, sentences: Sentences) -> (Contexts, Words, Sentences) {
        let cloned_sentences = sentences.clone();
        let mut contexts = Contexts::new();
        for sentence in cloned_sentences {
            let words = sentence.words.iter().map(|w| w.to_lowercase()).collect::<Vec<String>>();
            let mut buffer = Vec::<String>::new();
            for (_j, word) in words.iter().enumerate() {
                if !words.contains(word) {
                    buffer.clear();
                    continue;
                }

                let min_range = max(0 as i32, buffer.len() as i32 - self.config.window_size as i32);
                let max_range = buffer.len();
                let buffered_words = &buffer[(min_range as usize)..max_range as usize];
                for w in buffered_words {
                    let entry_1 = contexts.entry(word.to_string()).or_insert((
                        vec![w.to_string()],
                        Vec::<String>::new(),
                    ));
                    entry_1.0.push(w.to_string());
                    let entry_2 = contexts.entry(w.to_string()).or_insert((
                        Vec::<String>::new(),
                        vec![word.to_string()],
                    ));
                    entry_2.1.push(word.to_string());
                }
                buffer.push(word.to_string());
            }
        }

        (contexts, words, sentences)
    }

    fn feature_extraction(&mut self, contexts: Contexts, words: Words, sentences: Sentences) -> (Features, Contexts, Words, Sentences) {
        let tf = words.iter().map(|(_k,v)| v.len() ).collect::<Vec<usize>>();
        let tf_nsw = words.iter().filter_map(|(k,v)| {
            if !self.config.stopwords.contains(&k.to_owned()) {
                Some(v.len())
            } else {
                None
            }
        }).collect::<Vec<usize>>();

        let std_tf = stddev(tf_nsw.iter().map(|x| *x as f64));
        let mean_tf = mean(tf_nsw.iter().map(|x| *x as f64));
        let max_tf = *tf.iter().max().unwrap() as f64;

        let mut features = Features::new();
        for (key, ref word) in &words {

            let mut cand = YakeCandidate::default();
            cand.isstop = self.config.stopwords.contains(key) || key.len() < 3;
            cand.tf =  word.len() as f64;
            cand.tf_a = 0.0;
            cand.tf_u = 0.0;
            for occurrence in word.clone() {
                if occurrence.word.chars().all(|c| c.is_uppercase()) && occurrence.word.len() > 1 {
                     cand.tf_a += 1.0;
                }
                if occurrence.word.chars().nth(0).unwrap_or(' ').is_uppercase() && occurrence.shift != occurrence.shift_offset {
                    cand.tf_u += 1.0;
                }
            }

            cand.casing = cand.tf_a.max(cand.tf_u);
            cand.casing /= 1.0 + cand.tf.ln_1p();

            let sentence_ids = word.iter().map(|o| o.index).collect::<HashSet<usize>>();
            cand.position = (3.0 + median(sentence_ids.iter().map(|x| *x)).unwrap()).ln();
            cand.position = cand.position.ln();

            cand.frequency = cand.tf;
            cand.frequency /= mean_tf + std_tf;

            cand.wl = 0.0;

            let ctx = contexts.get(key).unwrap();
            let ctx_1_hash: HashSet<String> = HashSet::from_iter(ctx.clone().0);
            if ctx.0.len() > 0 {
                cand.wl = ctx_1_hash.len() as f64;
                cand.wl /=  ctx.0.len() as f64;
            }
            cand.pl = ctx_1_hash.len() as f64 / max_tf;

            cand.wr = 0.0;
            let ctx_2_hash: HashSet<String> = HashSet::from_iter(ctx.clone().1);
            if ctx.1.len() > 0 {
                cand.wr = ctx_2_hash.len() as f64;
                cand.wr /= ctx.1.len() as f64;
            }
            cand.pr = ctx_2_hash.len() as f64 / max_tf;

            cand.relatedness = 1.0;
            cand.relatedness += (cand.wr + cand.wl) * (cand.tf / max_tf);

            cand.different = sentence_ids.len() as f64;
            cand.different /= sentences.len() as f64;
            cand.weight = (cand.relatedness * cand.position) / (cand.casing + (cand.frequency / cand.relatedness) + ( cand.different / cand.relatedness));
        
            features.insert(key.to_string(), cand);
        }



        (features, contexts, words, sentences )
    }

    fn candidate_weighting(&mut self, features: Features, contexts: Contexts, candidates: Candidates) -> (HashMap<String, f64>,  HashMap<String, String>, HashMap<String, (Vec<String>, Vec<String>)>, HashMap<String, PreCandidate>) {
        let mut final_weights = HashMap::<String, f64>::new();
        let mut surface_to_lexical = HashMap::<String, String>::new();
        for (_k, v) in candidates.clone() {
                let lowercase_forms = v.surface_forms.iter().map(|w| w.join(" ").to_lowercase());
                for (idx, candidate) in lowercase_forms.clone().enumerate() {
                    let tf = lowercase_forms.clone().count() as f64;
                    let tokens = v.surface_forms[idx].iter().clone().map(|w| w.to_lowercase());
                    let mut prod_ = 1.0;
                    let mut sum_ = 0.0;
                    for (j, token) in tokens.clone().enumerate() {
                        let cand_value = match features.get_key_value(&token) {
                            Some(b) => b,
                            None => continue,
                        };
                        if cand_value.1.isstop  {
                            let term_stop = token;
                            let mut prob_t1 = 0.0;
                            let mut prob_t2 = 0.0;
                            if j - 1 > 0 {
                                let term_left = tokens.clone().nth(j-1).unwrap();
                                prob_t1 = contexts.get(&term_left).unwrap().1.iter().filter(|w| **w == term_stop).count() as f64 / features.get(&term_left).unwrap().tf;
                            }
                            if j + 1 < tokens.len() {
                                let term_right = tokens.clone().nth(j+1).unwrap();
                                prob_t2 = contexts.get(&term_stop).unwrap().0.iter().filter(|w| **w == term_right).count() as f64 / features.get(&term_right).unwrap().tf;
                            }

                            let prob = prob_t1 * prob_t2;
                            prod_ *= 1.0 + (1.0 - prob );
                            sum_ -= 1.0 - prob;
                        } else {
                            prod_ *= cand_value.1.weight;
                            sum_  += cand_value.1.weight;
                        }
                    }
                    if sum_ == -1.0 {
                        sum_ = 0.999999999;
                    }

                    let weight = prod_ / tf * (1.0 + sum_);

                    final_weights.insert(candidate.to_string(), weight);
                    surface_to_lexical.insert(candidate.to_string(), v.lexical_form.join(" "));
                } 
        }

        (final_weights, surface_to_lexical, contexts, candidates)
    }

    fn is_redundant(&mut self, cand: String, prev: Vec<String>) -> bool {
        for prev_cand in prev {
            let dist = levenshtein::Levenshtein::ratio(cand.to_owned(), prev_cand);
            if dist > self.config.dedupe_lim {
                return true;
            }
        }

        false
    } 

    fn is_alphanum(&mut self, mut word: String, valid_punctuation_marks: Option<String>) -> bool {
        let default_valid_punctuation_marks = valid_punctuation_marks.unwrap_or("-".to_owned());
        for punct in default_valid_punctuation_marks.split("") {
            word = word.replace(punct, "");
        }
        word.chars().all(|c| c.is_alphanumeric())
    }

    fn candidate_filtering(&mut self, mut candidates: Candidates ,minimum_length: Option<usize>, minimum_word_size: Option<usize>, valid_punctuation_marks: Option<String>, maximum_word_number: Option<usize>, only_alphanum: Option<bool>) -> Candidates {
        let default_minimum_length = minimum_length.unwrap_or(3);
        let default_minimum_word_size = minimum_word_size.unwrap_or(2);
        let default_maximum_word_number = maximum_word_number.unwrap_or(5);
        let default_only_alphanum = only_alphanum.unwrap_or(false);
        let default_valid_punctuation_marks = valid_punctuation_marks.unwrap_or("-".to_owned());


        for (k, v) in candidates.clone() {
            //get the words from the first occurring surface form
            let words = HashSet::from_iter(v.surface_forms[0].iter().map(|w| w.to_lowercase()));
            if words.intersection(&self.config.stopwords).count() > 0 {
                candidates.remove_entry(&k);
            }
            if words.clone().iter().any(|w| w.parse::<f64>().is_ok()) {
                candidates.remove_entry(&k);
            }
            if words.clone().iter().any(|w| HashSet::from_iter(vec![w.to_owned()]).is_subset(&self.config.punctuation)) {
                candidates.remove_entry(&k);
            }
            if words.clone().iter().map(|w| w.to_owned()).collect::<Vec<String>>().join("").len() < default_minimum_length {
                candidates.remove_entry(&k);
            }; 
            if words.clone().iter().map(|w| w.len()).min().unwrap() < default_minimum_word_size {
                candidates.remove_entry(&k);
            }
            if v.lexical_form.len() > default_maximum_word_number {
                candidates.remove_entry(&k);
            } 
            if default_only_alphanum && candidates.contains_key(&k) {
                if words.clone().iter().any(|w| !self.is_alphanum(w.to_owned(), Some(default_valid_punctuation_marks.to_owned()))) {
                    candidates.remove_entry(&k);
                }
            }
        }

        candidates
    }

    fn ngram_selection(&mut self, n: usize, sentences: Sentences) -> (Candidates, Sentences)  {
        let mut candidates = HashMap::<String, PreCandidate>::new();
        for (idx, sentence) in sentences.iter().enumerate() {
            let skip = min(n, sentence.length);
            let shift = sentences[0..idx].iter().map(|s| s.length).sum::<usize>();

            for j in 0..sentence.length {
                for k in j+1..min(j + 1 + skip, sentence.length + 1) {

                    let words = sentence.words[j..k].to_vec();                
                    let stems = sentence.stems[j..k].to_vec();
                    let sentence_id = idx;
                    let offset = j + shift;
                    let lexical_form = stems.join(" ");
                    let candidate = candidates.get_mut(lexical_form.as_str());
                    if candidate.is_none() {
                        candidates.insert(lexical_form.clone(), PreCandidate {
                            lexical_form: stems,
                            surface_forms: vec![words],
                            sentence_ids: vec![sentence_id],
                            offsets: vec![offset],
                        });
                    } else {
                        let candidate = candidate.unwrap();
                        candidate.surface_forms.push(words);
                        candidate.sentence_ids.push(sentence_id);
                        candidate.offsets.push(offset);
                        candidate.lexical_form = stems;
                    }
                }
            }
        }

        

        (candidates, sentences)
    }
    
} 