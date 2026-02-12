use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, Clone, StructOpt)]
pub struct Cli {
    #[structopt(short, long, parse(from_os_str))]
    pub input: PathBuf,

    #[structopt(long, default_value = "main")]
    pub entry: String,

    #[structopt(long = "expect-i32")]
    pub expect_i32: Option<i32>,

    #[structopt(long, default_value = "1")]
    pub iterations: usize,

    #[structopt(long, default_value = "0")]
    pub warmup: usize,

    #[structopt(long)]
    pub quiet: bool,

    #[structopt(long)]
    pub json: bool,
}
