use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct RunReport {
    pub expected: Option<i32>,
    pub actual: i32,
    pub pass: bool,
    pub iterations: usize,
    pub warmup: usize,
    pub avg_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub print_count: u64,
    pub print_hash: u64,
}

impl RunReport {
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        let status = if self.pass { "PASS" } else { "FAIL" };
        let expected = self
            .expected
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".to_string());
        let _ = write!(
            out,
            "RESULT status={} expected={} actual={} iterations={} warmup={} ms_avg={:.6} ms_min={:.6} ms_max={:.6} print_count={} print_hash=0x{:016x}",
            status,
            expected,
            self.actual,
            self.iterations,
            self.warmup,
            self.avg_ms,
            self.min_ms,
            self.max_ms,
            self.print_count,
            self.print_hash
        );
        out
    }

    pub fn to_json(&self) -> String {
        let expected = self
            .expected
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string());
        format!(
            "{{\"pass\":{},\"expected\":{},\"actual\":{},\"iterations\":{},\"warmup\":{},\"ms_avg\":{:.6},\"ms_min\":{:.6},\"ms_max\":{:.6},\"print_count\":{},\"print_hash\":\"0x{:016x}\"}}",
            if self.pass { "true" } else { "false" },
            expected,
            self.actual,
            self.iterations,
            self.warmup,
            self.avg_ms,
            self.min_ms,
            self.max_ms,
            self.print_count,
            self.print_hash
        )
    }
}
