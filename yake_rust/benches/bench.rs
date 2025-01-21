use std::hint::black_box;
use std::io::{Cursor, Read};

use divan::{AllocProfiler, Bencher};
use yake_rust::{Config, StopWords, Yake};
use zip::ZipArchive;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    // Run registered benchmarks.
    divan::main();
}

#[divan::bench(ignore)]
fn text_100kb(bencher: Bencher) {
    let text = include_str!("100kb.txt");
    let stopwords = StopWords::predefined("en").unwrap();
    let config = Config { remove_duplicates: false, ..Default::default() };
    let yake = Yake::new(stopwords, config);

    bencher.bench_local(move || {
        black_box(yake.get_n_best(black_box(text), None));
    });
}

#[divan::bench(min_time = 10)]
fn text_3kb(bencher: Bencher) {
    let text = include_str!("3kb.txt");
    let stopwords = StopWords::predefined("en").unwrap();
    let config = Config { remove_duplicates: false, ..Default::default() };
    let yake = Yake::new(stopwords, config);

    bencher.bench_local(move || {
        black_box(yake.get_n_best(black_box(text), None));
    });
}

fn _read_zip(zip: &[u8]) -> Vec<String> {
    let mut zip = ZipArchive::new(Cursor::new(zip)).unwrap();
    let mut list = Vec::with_capacity(zip.len());

    for idx in 0..zip.len() {
        let mut file = zip.by_index(idx).unwrap();
        let mut text = String::new();
        file.read_to_string(&mut text).unwrap();
        list.push(text);
    }

    list
}
