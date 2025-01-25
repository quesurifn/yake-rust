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
```python
>>>from yake_rust import Yake
>>>yake = Yake(language="en")
>>>yake.get_n_best("News from the inauguration!", n=1)
[('inauguration', 0.15831692877998726)]
```
