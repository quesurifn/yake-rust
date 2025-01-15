use cli::{parse_cli, ParsedCli};
use yake_rust::Yake;

mod cli;

fn main() {
    let ParsedCli { language, input, config, top, verbose: _ } = parse_cli();

    let now = std::time::Instant::now();

    let keywords = Yake::new(language, config).get_n_best(&input, top);

    println!("{:?}", keywords);
    println!("Elapsed: {:.2?}", now.elapsed());
}
