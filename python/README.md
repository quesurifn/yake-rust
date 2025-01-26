# yake-rust

This package contains the [yake-rust](https://crates.io/crates/yake-rust) crate with python bindings.

Yake is a language agnostic statistical keyword extractor weighing several factors such as acronyms, position in
paragraph, capitalization, how many sentences the keyword appears in, stopwords, punctuation and more. For details,
see these papers: [brief](https://repositorio.inesctec.pt/server/api/core/bitstreams/ef121a01-a0a6-4be8-945d-3324a58fc944/content),
[extended](https://doi.org/10.1016/j.ins.2019.09.013).

This crate is ported and is as close as possible in results to the [reference implementation](https://github.com/LIAAD/yake/).
The input text is split into sentences and tokens via the [segtok](https://github.com/xamgore/segtok) crate.

By implementing the keyword extraction in rust instead of native python, *yake-rust* has the advantage of
- **speed** - in tests, it typically outperforms the reference implementation.
- **concurrency** - *yake_rust* releases the GIL so that multiple threads can perform keyword extraction
concurrently on multicore systems.

Furthermore, *yake-rust* is fully **typed**.

## Installation

For example, to install it with `pip`:
```bash
pip install yake-rust
```

## Usage

Create an instance of `yake_rust.Yake` with the desired configuration
and call its `get_n_best` method to get the `n` best keywords from a text.
(Note that `n` is a keyword-only argument for the purpose of clarity.)
```python
>>>from yake_rust import Yake
>>>yake = Yake(language="en")
>>>yake.get_n_best("News from the inauguration!", n=1)
[('inauguration', 0.15831692877998726)]
```

The `yake_rust.Yake` constructor requires one (and only one) of two mandatory keyword-only arguments,
- `language` - [ISO 639](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) two-letter
abbreviation for a language, e.g., `language="en"` for English.
- `stopwords` - `set[str]` of custom stopwords to use.
If `language` is used, a default set of stopwords for that language will be used.

In addition, it has the following keyword-only arguments for customizability:
- `ngrams` - `int` specifying the maximum number of ngrams per keyword.
- `punctuation` - `set[str]` with punctuation symbols.
- `window_size` - `int` determining the size of the "window" around a word when
considering how it appears in context.
- `remove_duplicates` - `bool` saying whether to remove keywords which are considered
duplicates of other keywords, as determined by the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
and the deduplication threshold.
- `deduplication threshold` - `float` which determines the threshold of considering two keywords
to be duplicates of each other. Does nothing if `remove_duplicates` is `False`. 
- `strict_capital` - `bool` that when `True` (default), consider the casing of a term
by counting capitalized terms without intermediate uppercase letters. Thus, `Paypal` is counted while `PayPal` is not.
The [reference implementation](https://github.com/LIAAD/yake) sticks with `True`.
- `only_alphanumeric_and_hyphen` - `bool` that can be turned on the drop keywords that contain
characters other than alphanumerical characters and hyphens.
- `minimum_chars` - `int` for the minimum number of characters in a keyword; shorter keywords
will be dropped. By default, there is no such limit.

Leaving the defaults should bring you as close as possible to the
[reference implementation](https://github.com/LIAAD/yake). 
For more information and precise defaults, see the [rust crate](https://crates.io/crates/yake-rust).
