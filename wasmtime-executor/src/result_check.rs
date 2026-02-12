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
            "RESULT status={} expected={} actual={} iterations={} warmup={} ms_avg={:.6} ms_min={:.6} ms_max={:.6}",
            status,
            expected,
            self.actual,
            self.iterations,
            self.warmup,
            self.avg_ms,
            self.min_ms,
            self.max_ms
        );
        out
    }

    pub fn to_json(&self) -> String {
        let expected = self
            .expected
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string());
        format!(
            "{{\"pass\":{},\"expected\":{},\"actual\":{},\"iterations\":{},\"warmup\":{},\"ms_avg\":{:.6},\"ms_min\":{:.6},\"ms_max\":{:.6}}}",
            if self.pass { "true" } else { "false" },
            expected,
            self.actual,
            self.iterations,
            self.warmup,
            self.avg_ms,
            self.min_ms,
            self.max_ms
        )
    }
}
