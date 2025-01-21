use cli::{parse_cli, ParsedCli};
use prettytable::{format, row, Table};
use yake_rust::{ResultItem, Yake};

mod cli;

fn main() {
    let ParsedCli { language, json, input, config, top, verbose } = parse_cli();

    let now = std::time::Instant::now();

    let keywords = Yake::new(language, config).get_n_best(&input, top);

    output_keywords(&keywords, json, verbose);
    if verbose {
        eprintln!("Elapsed: {:.2?}", now.elapsed());
    }
}

fn output_keywords(keywords: &[ResultItem], json: bool, verbose: bool) {
    match (json, verbose) {
        (true, _) => {
            output_keywords_json(&keywords);
        }
        (false, true) => {
            output_keywords_verbose(&keywords);
        }
        (false, false) => {
            output_keywords_simple(&keywords);
        }
    }
}

fn output_keywords_verbose(keywords: &[ResultItem]) {
    let mut table = Table::new();
    table.set_titles(row!["keyword", "raw", "score"]);
    for keyword in keywords {
        table.add_row(row![keyword.keyword, keyword.raw, format!("{:.4}", keyword.score)]);
    }
    table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
    table.printstd()
}

fn output_keywords_simple(keywords: &[ResultItem]) {
    let mut table = Table::new();
    table.set_titles(row!["keyword"]);
    for keyword in keywords {
        table.add_row(row![keyword.keyword]);
    }
    table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
    table.printstd()
}

fn output_keywords_json(keywords: &[ResultItem]) {
    match serde_json::to_string(&keywords) {
        Ok(str) => {
            println!("{}", str)
        }
        Err(e) => {
            eprintln!("Unexpected error happened while trying to serialize result to json : {:?}", e);
            std::process::exit(exit_code::SOFTWARE_ERROR)
        }
    }
}
