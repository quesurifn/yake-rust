# YAKE (Yet Another Keyword Extractor) [![](https://img.shields.io/crates/v/yake-rust.svg)](https://crates.io/crates/yake-rust) [![](https://docs.rs/yake-rust/badge.svg)](https://docs.rs/yake-rust/)

Yake is a language agnostic statistical keyword extractor weighing several factors such as acronyms, position in
paragraph, capitalization, how many sentences the keyword appears in, stopwords, punctuation and more. Details are in
these papers: [brief](https://repositorio.inesctec.pt/server/api/core/bitstreams/ef121a01-a0a6-4be8-945d-3324a58fc944/content),
[extended](https://doi.org/10.1016/j.ins.2019.09.013).

This crate is ported and is as close as possible to the [reference implementation](https://github.com/LIAAD/yake/).
The input text is split into sentences and tokens via the [segtok](https://github.com/xamgore/segtok) crate.

## How it works

For Yake ✨keyphrase✨ is an n-gram (1-, 2-, 3-) not starting nor ending in a stopword, not having numbers and punctuation inside, without long and short terms, etc.

Yake assigns an importance score to each term in the text. The lower the score, the more important the term.

Eventually, the most important terms:
- occur more frequently
- occur mostly at the beginning of the text
- occur in many different sentences
- prefer being Capitalized or UPPERCASED
- prefer having the same neighbour terms

✨Keyphrases✨ are ranked in order of importance (most important first).

Duplicates are then detected by [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) and removed.

## Example

```rust
use yake_rust::{get_n_best, Config, StopWords};

fn main() {
    let text = include_str!("input.txt");

    let config = Config { ngrams: 3, ..Config::default() };
    let ignored = StopWords::predefined("en").unwrap();
    let keywords = get_n_best(10, &text, &ignored, &config);

    println!("{:?}", keywords);
}
```

## Features
By default, stopwords for all languages are included. However, you can choose to include only specific ones by doing the following:

```toml
[dependencies]
yake-rust = { version = "*", default-features = false, features = ["en", "de"] }
```
