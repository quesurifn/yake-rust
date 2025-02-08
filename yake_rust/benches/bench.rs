use std::hint::black_box;
use std::io::{Cursor, Read};

use divan::Bencher;
use yake_rust::{get_n_best, Config, StopWords};
use zip::ZipArchive;

// #[global_allocator]
// static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    // Run registered benchmarks.
    divan::main();
}

#[divan::bench(min_time = 60)]
fn text_100kb(bencher: Bencher) {
    let text = include_str!("100kb.txt");
    let stopwords = StopWords::predefined("en").unwrap();
    let config = Config { remove_duplicates: false, ..Default::default() };
    segtok::init();

    bencher.bench_local(move || {
        black_box(get_n_best(usize::MAX, black_box(text), black_box(&stopwords), black_box(&config)));
    });
}

#[divan::bench(min_time = 60)]
fn text_3kb(bencher: Bencher) {
    let text = include_str!("3kb.txt");
    let stopwords = StopWords::predefined("en").unwrap();
    let config = Config { remove_duplicates: false, ..Default::default() };
    segtok::init();

    bencher.bench_local(move || {
        black_box(get_n_best(usize::MAX, black_box(text), black_box(&stopwords), black_box(&config)));
    });
}

#[divan::bench(min_time = 10)]
fn text_170b(bencher: Bencher) {
    let text = "Do you like headphones? \
            Starting this Saturday, we will be kicking off a huge sale of headphones! \
            If you need headphones, we've got you covered!";
    let stopwords = StopWords::predefined("en").unwrap();
    let config = Config { remove_duplicates: false, ..Default::default() };
    segtok::init();

    bencher.bench_local(move || {
        black_box(get_n_best(usize::MAX, black_box(text), black_box(&stopwords), black_box(&config)));
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
