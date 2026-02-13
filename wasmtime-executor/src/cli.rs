use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintMode {
    Normal,
    Hash,
}

impl PrintMode {
    pub fn is_hash(self) -> bool {
        matches!(self, Self::Hash)
    }
}

impl std::str::FromStr for PrintMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "normal" => Ok(Self::Normal),
            "hash" => Ok(Self::Hash),
            _ => Err(format!(
                "invalid print mode '{}': expected 'normal' or 'hash'",
                value
            )),
        }
    }
}

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

    #[structopt(
        long,
        default_value = "normal",
        possible_values = &["normal", "hash"]
    )]
    pub print_mode: PrintMode,

    #[structopt(long, default_value = "14695981039346656037")]
    pub print_hash_seed: u64,
}
