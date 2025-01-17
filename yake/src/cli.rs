use std::{path::PathBuf, sync::LazyLock};

use clap::error::ErrorKind;
use clap::{command, Args};
use clap::{CommandFactory, Parser};
use yake_rust::{Config, StopWords};

static DEFAULT_CONFIG: LazyLock<Config> = LazyLock::new(Config::default);

#[derive(Args)]
#[group(required = true, multiple = false)]
struct Input {
    // -ti, --text_input TEXT
    /// Input text
    #[arg(
        conflicts_with = "input_file",
        long,
        help = "Input text, SURROUNDED by single quotes(')",
        value_name = "TEXT"
    )]
    text_input: Option<String>,

    // -i, --input_file TEXT
    /// Input file
    #[arg(conflicts_with = "text_input", short, long, help = "Input file", value_name = "FILE")]
    input_file: Option<PathBuf>,
}

// TODO
// -df, --dedup-func [leve|jaro|seqm]
// Deduplication function.

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(flatten)]
    input: Input,

    // -n, --ngram-size INTEGER
    /// Max size of the ngram
    #[arg(short, long, default_value_t = DEFAULT_CONFIG.ngrams, help = "Max size of the ngram", value_name = "INTEGER")]
    ngram_size: usize,

    // -dl, --dedup-lim FLOAT
    /// Deduplication limiter
    #[arg(long, value_parser = parse_dedup, default_value_t = DEFAULT_CONFIG.deduplication_threshold, help = "Deduplication limiter", value_name = "FLOAT")]
    dedup_lim: f64,

    // -ws, --window-size INTEGER
    /// Window size
    #[arg(long, default_value_t = DEFAULT_CONFIG.window_size, help = "Window size", value_name = "INTEGER")]
    window_size: usize,

    // -t, --top INTEGER
    /// Number of keyphrases to extract
    #[arg(short, long, help = "Number of keyphrases to extract", value_name = "INTEGER")]
    top: Option<usize>,

    // -v, --verbose
    /// Gets detailed information (such as the score)
    #[arg(short, long, help = "Gets detailed information (such as the score)")]
    verbose: bool,

    // // --help
    // /// Show this message and exit
    // #[arg(short, long)]
    // help: bool,

    // -l, --language TEXT
    /// Language
    #[arg(short, long, default_value= "en", value_parser = parse_language, help = "Language", value_name = "TEXT")]
    language: StopWords,

    #[arg(long, help = "Dump output as JSON")]
    json: bool,
}

fn parse_language(cli_language: &str) -> Result<StopWords, String> {
    StopWords::predefined(cli_language)
        .ok_or_else(|| format!("Could not find language {}, did you enable this feature?", cli_language))
}

fn parse_dedup(cli_dedup_lim: &str) -> Result<f64, String> {
    match cli_dedup_lim.parse::<f64>() {
        Ok(value @ 0f64..=1f64) => Ok(value),
        Ok(value) => Err(format!("{} is not in the 0..=1", value)),
        Err(_) => Err("invalid digit found in string".into()),
    }
}

pub struct ParsedCli {
    pub config: Config,
    pub language: StopWords,
    pub input: String,
    pub json: bool,
    pub top: Option<usize>,
    pub verbose: bool,
}

pub fn parse_cli() -> ParsedCli {
    let cli = Cli::parse();

    let input = match (cli.input.text_input, cli.input.input_file) {
        (None, None) | (Some(_), Some(_)) => {
            panic!("clap should ensure that either text-input or input-file is specified")
        }
        (None, Some(path_to_file)) => match std::fs::read_to_string(&path_to_file) {
            Ok(text) => text,
            Err(err) => {
                Cli::command()
                    .error(
                        ErrorKind::ValueValidation,
                        format!("Error reading file `{}`: {:?}", path_to_file.display(), err),
                    )
                    .exit();
            }
        },
        (Some(text), None) => text,
    };

    ParsedCli {
        config: Config {
            ngrams: cli.ngram_size,
            window_size: cli.window_size,
            deduplication_threshold: cli.dedup_lim,
            ..Config::default()
        },
        language: cli.language,
        input,
        json: cli.json,
        verbose: cli.verbose,
        top: cli.top,
    }
}
