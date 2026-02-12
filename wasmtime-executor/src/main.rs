mod cli;
mod host_env;
mod result_check;
mod runtime;

use cli::Cli;
use structopt::StructOpt;

fn main() {
    let cli = Cli::from_args();
    let quiet = cli.quiet;
    let json = cli.json;

    match runtime::run(&cli) {
        Ok(report) => {
            if json {
                println!("{}", report.to_json());
            } else if !quiet {
                println!("{}", report.to_text());
            }
            std::process::exit(0);
        }
        Err(err) => {
            eprintln!("ERROR: {}", err);
            std::process::exit(err.exit_code());
        }
    }
}
