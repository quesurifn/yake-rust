use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use yake_rust::{Config, ResultItem, StopWords, Yake};

#[derive(Clone, Copy, ValueEnum)]
enum DedupFunction {
    Level,
    Jaro,
    SeqM,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    // -ti, --text_input TEXT          Input text, SURROUNDED by single quotes(')
    #[arg(conflicts_with = "input_file", long)]
    text_input: String,

    // -i, --input_file TEXT           Input file
    #[arg(conflicts_with = "text_input", short, long)]
    input_file: PathBuf,

    // -n, --ngram-size INTEGER        Max size of the ngram.
    #[arg(short, long)]
    ngram_size: usize,

    // -df, --dedup-func [leve|jaro|seqm]
    // 								Deduplication function.
    #[arg(value_enum, long)]
    dedup_func: DedupFunction,

    // -dl, --dedup-lim FLOAT          Deduplication limiar.
    #[arg(long, value_parser = dedup_parser)]
    dedup_lim: f64,

    // -ws, --window-size INTEGER      Window size.
    #[arg(long)]
    window_size: usize,

    // -t, --top INTEGER               Number of keyphrases to extract
    #[arg(short, long, default_value = "10")]
    top: u32,

    // -v, --verbose			Gets detailed information (such as the score)
    #[arg(short, long)]
    verbose: bool,

    // --help                          Show this message and exit.
    // #[arg(short, long)]
    // help: bool,

    // -l, --language TEXT             Language
    #[arg(short, long, default_value= "en", value_parser = StopWords::predefined)]
    language: StopWords,

    #[arg(long, action)]
    json: bool,
}

fn dedup_parser(cli_dedup_lim: &str) -> Result<f64, String> {
    match cli_dedup_lim.parse::<f64>() {
        Ok(value @ 0f64..=1f64) => Ok(value),
        Ok(value) => Err(format!("The value {} does not fall between the 0..=1 range", value)),
        Err(value) => Err(format!("Could not parse {} as f64", value)),
    }
}

fn main() {
    let cli = Cli::parse();

    let config: Config = Config {
        ngrams: cli.ngram_size,
        window_size: cli.window_size,
        deduplication_threshold: cli.dedup_lim,
        ..Default::default()
    };

    let text = r#"
    Google is acquiring data science community Kaggle. Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning
    competitions. Details about the transaction remain somewhat vague, but given that Google is hosting its Cloud
    Next conference in San Francisco this week, the official announcement could come as early as tomorrow.
    Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening.
    Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data scientists on its platform,
    was founded by Goldbloom  and Ben Hamner in 2010.
    The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank,
    it has managed to stay well ahead of them by focusing on its specific niche.
    The service is basically the de facto home for running data science and machine learning competitions.
    With Kaggle, Google is buying one of the largest and most active communities for data scientists - and with that,
    it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow
    and other projects). Kaggle has a bit of a history with Google, too, but that's pretty recent. Earlier this month,
    Google and Kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos.
    That competition had some deep integrations with the Google Cloud Platform, too. Our understanding is that Google
    will keep the service running - likely under its current name. While the acquisition is probably more about
    Kaggle's community than technology, Kaggle did build some interesting tools for hosting its competition
    and 'kernels', too. On Kaggle, kernels are basically the source code for analyzing data sets and developers can
    share this code on the platform (the company previously called them 'scripts').
    Like similar competition-centric sites, Kaggle also runs a job board, too. It's unclear what Google will do with
    that part of the service. According to Crunchbase, Kaggle raised $12.5 million (though PitchBook says it's $12.75)
    since its   launch in 2010. Investors in Kaggle include Index Ventures, SV Angel, Max Levchin, Naval Ravikant,
    Google chief economist Hal Varian, Khosla Ventures and Yuri Milner
    "#;

    let now = std::time::Instant::now();

    let keywords = Yake::new(StopWords::predefined("en").unwrap(), config).get_n_best(text, Some(10));
    output_keywords(&keywords, cli.json);
    eprintln!("Elapsed: {:.2?}", now.elapsed());
}

fn output_keywords(keywords: &Vec<ResultItem>, json: bool) {
    if json {
        match serde_json::to_string(&keywords) {
            Ok(str) => {
                println!("{}", str)
            }
            Err(e) => {
                eprintln!("Unexpected error happened while trying to serialize result to json : {:?}", e);
                std::process::exit(exit_code::SOFTWARE_ERROR)
            }
        }
    } else {
        println!("{:?}", keywords);
    }
}
