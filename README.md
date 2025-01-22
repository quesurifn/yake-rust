# YAKE (Yet Another Keyword Extractor) [![](https://img.shields.io/crates/v/yake-rust.svg)](https://crates.io/crates/yake-rust) [![](https://docs.rs/yake-rust/badge.svg)](https://docs.rs/yake-rust/)

Yake is a language agnostic statistical keyword extractor weighing several factors such as acronyms, position in
paragraph, capitalization, how many sentences the keyword appears in, stopwords, punctuation and more. Details are in
these papers: [brief](https://repositorio.inesctec.pt/server/api/core/bitstreams/ef121a01-a0a6-4be8-945d-3324a58fc944/content),
[extended](https://doi.org/10.1016/j.ins.2019.09.013).

This crate is ported and is as close as possible to the [reference implementation](https://github.com/LIAAD/yake/).
The input text is split into sentences and tokens via the [segtok](https://github.com/xamgore/segtok) crate.

## How it works

For Yake ✨keyphrase✨ is an n-gram (1-, 2-, 3-) not starting nor ending in a stopword, not having numbers and punctuation inside, without long and short terms, etc.

Yake assigns an importance score $\mathbf{T}\_\text{Score}$ to each term in the text. The lower the score, the more important the term.

$$
\begin{flalign}
& \mathbf{T}\_\text{NormedFreq} &&=&& \frac{ \text{frequency of } \text{term }\mathbf{T} }{\text{mean}\\,\bigl(\\,\text{frequency of term }\mathbb{T} | \forall\\, \mathbb{T} \notin SW\\, \bigr) + 1 \sigma }  \\
& \mathbf{T}\_\text{Casing} &&=&& \frac{\text{frequency of term }\mathbf{T}\text{ written as 'TERM'} }{\ln \bigl(\\, \text{frequency of term }\mathbf{T}\\,\bigr)}  \\
& \mathbf{T}\_\text{\\%Sentences} &&=&& \dfrac{ |S| }{ \text{\\# of sentences } }, \text{ where } S = \\{\\, i\\, |\\, \text{term }\mathbf{T} \in \text{sentence}_i\\, \\} \\
& \mathbf{T}\_\text{Relatedness} &&=&& 1 + (\mathbf{T}\_\text{LDiversity} + \mathbf{T}\_\text{RDiversity}) * \frac{ \text{frequency of } \text{term }\mathbf{T} }{\max\bigl(\text{ frequency of term }\mathbb{T} | \forall\\, \mathbb{T} \\, \bigr) } \\
& \mathbf{T}\_\text{Position} &&=&& \overset{\ } \ln ( \ln (\\, 3 + \text{median}(S) \\,) ), \text{ where } S = \\{\\, i\\, |\\, \text{term }\mathbf{T} \in \text{sentence}_i\\, \\} \\
& \mathbf{T}\_\text{Score} &&=&& \dfrac{ {\mathbf{T}\_\text{Relatedness}}^2 * \mathbf{T}\_\text{Position} \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad }{ \mathbf{T}\_\text{Relatedness} \ * \mathbf{T}\_\text{Casing} + \mathbf{T}\_\text{NormedFreq} + \mathbf{T}\_\text{\\%Sentences} } \\
& \mathcal{K}\_\text{Score} &&=&& \frac{\prod\_{\mathbf{T} \in \mathcal{K}} \mathbf{T}\_\text{Score}}{\text{frequency of keyword }\mathcal{K} * (1 + \sum\_{\mathbf{T} \in \mathcal{K}} \mathbf{T}\_\text{Score})}
\end{flalign}
$$

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
use yake_rust::*;

fn main() {
    let text = include_str!("input.txt");

    let keywords =
        Yake::new(StopWords::predefined("en").unwrap(), Config::default())
            .get_n_best(text, Some(10));

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

| score | top 10 keywords            |
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
