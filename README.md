# YAKE (Yet Another Keyword Extractor) [![](https://img.shields.io/crates/v/yake-rust.svg)](https://crates.io/crates/yake-rust) [![](https://docs.rs/yake-rust/badge.svg)](https://docs.rs/yake-rust/)

Yake is a statistical keyword extractor. It weighs several factors such as acronyms, position in
paragraph, capitalization, how many sentences the keyword appears in, stopwords, punctuation and more.

## How it works

For Yake ✨keyphrase✨ is an n-gram (1-, 2-, 3-) not starting nor ending in a stopword, not having numbers and punctuation inside, without long and short terms, etc.

The input text is split into sentences and terms via the [segtok](https://github.com/xamgore/segtok) crate.
Yake assigns an importance score to each term in the text.

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

<details><summary>input.txt</summary>

> **Google** is **acquiring** **data science** community **Kaggle**. Sources tell us that **Google** is acquiring **Kaggle**,
> a platform that hosts **data science** and machine learning competitions. Details about the transaction remain somewhat
> vague, but given that **Google** is hosting its Cloud Next conference in **San Francisco** this week, the official announcement could come as early as tomorrow.
> Reached by phone, **Kaggle** co-founder **CEO Anthony Goldbloom** **declined** to deny that the acquisition is happening.
> **Google** itself declined 'to comment on rumors'. **Kaggle**, which has about half a million **data** scientists on its platform,
> was founded by Goldbloom and Ben Hamner in 2010.
> The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank,
> it has managed to stay well ahead of them by focusing on its specific niche.
> The service is basically the de facto home for running **data science** and machine learning competitions.
> With **Kaggle**, **Google** is buying one of the largest and most active communities for **data** scientists - and with that,
> it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow
> and other projects). **Kaggle** has a bit of a history with **Google**, too, but that's pretty recent. Earlier this month,
> **Google** and **Kaggle** teamed up to host a $100,000 machine learning competition around classifying YouTube videos.
> That competition had some deep integrations with the **Google** **Cloud Platform**, too. Our understanding is that **Google**
> will keep the service running - likely under its current name. While the acquisition is probably more about
> **Kaggle**'s community than technology, **Kaggle** did build some interesting tools for hosting its competition
> and 'kernels', too. On **Kaggle**, kernels are basically the source code for analyzing **data** sets and developers can
> share this code on the platform (the company previously called them 'scripts').
> Like similar competition-centric sites, **Kaggle** also runs a job board, too. It's unclear what **Google** will do with
> that part of the service. According to Crunchbase, **Kaggle** raised \$12.5 million (though PitchBook says it's \$12.75)
> since its launch in 2010. Investors in **Kaggle** include Index Ventures, SV Angel, Max Levchin, Naval Ravikant,
> **Google** chief economist Hal Varian, Khosla Ventures and Yuri Milner
</details>

| Score | Top 10 keywords            |
|------:|:---------------------------|
| 0.025 | Google                     |
| 0.027 | Kaggle                     |
| 0.048 | CEO Anthony Goldbloom      |
| 0.055 | data science               |
| 0.060 | acquiring data science     |
| 0.075 | Google Cloud Platform      |
| 0.080 | data                       |
| 0.091 | San Francisco              |
| 0.097 | Anthony Goldbloom declined |
| 0.098 | science                    |


## Features
By default, stopwords for all languages are included. However, you can choose to include only specific ones:

```toml
[dependencies]
yake-rust = { version = "*", default-features = false, features = ["en", "de"] }
```

## Credits

This Rust crate implements the algorithm described in papers
([brief](https://web.archive.org/web/20240418035141/https://repositorio.inesctec.pt/server/api/core/bitstreams/ef121a01-a0a6-4be8-945d-3324a58fc944/content),
[extended](https://doi.org/10.1016/j.ins.2019.09.013)) and was inspired by
the Python package [yake](https://github.com/LIAAD/yake/), originally developed by
Ricardo Campos, Vitor Mangaravite, Arian Pasquali, Alípio Jorge.

- [Kyle Fahey](https://github.com/quesurifn) — implemented the first draft
- [Anton Vikström](https://github.com/bunny-therapist) — reached 1-1 score alignment with the original package
- [Igor Strebz](https://github.com/xamgore) — rewritten and heavily optimized the code
