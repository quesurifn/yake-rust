use std::fs::File;
use std::io::{BufReader, Read};

use yake_rust::{Config, StopWords, Yake};
use zip::ZipArchive;

#[test]
#[ignore = "run manually with cargo test --all -- --include-ignored --nocapture"]
fn run_through_dataset_files() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/datasets"))?;

    let files: &[(&str, &str)] = &[
        ("110-PT-BN-KP", "pt"),
        ("500N-KPCrowd-v1.1", "en"),
        ("Inspec", "en"),
        ("Krapivin2009", "en"),
        ("Nguyen2007", "en"),
        ("PubMed", "en"),
        ("Schutz2008", "en"),
        ("SemEval2010", "en"),
        ("SemEval2017", "en"),
        ("WikiNews", "fr"),
        ("cacic", "es"),
        ("citeulike180", "en"),
        ("fao30", "en"),
        ("fao780", "en"),
        ("kdd", "en"),
        ("pak2018", "pl"),
        ("theses100", "en"),
        ("wicc", "es"),
        ("wiki20", "en"),
        ("www", "en"),
    ];

    for &(filename, lang) in files {
        println!("\n{filename}:");
        let filename = format!("{filename}.zip");
        let mut zip = ZipArchive::new(BufReader::new(File::open(filename)?))?;

        for idx in 0..zip.len() {
            let mut file = zip.by_index(idx)?;
            if file.is_dir() || !file.name().contains("docsutf8") {
                continue;
            }

            let mut text = String::new();
            file.read_to_string(&mut text).unwrap();

            let result = std::panic::catch_unwind(move || {
                let stopwords = StopWords::predefined(lang).unwrap();
                let yake = Yake::new(stopwords, Config::default());
                let _ = yake.get_n_best(&text, 10);
            });

            if result.is_err() {
                println!("{}", file.name());
            }
        }
    }

    Ok(())
}
