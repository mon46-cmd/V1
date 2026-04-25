//! Regime kernels: Hurst exponent (R/S), autocorrelation, variance ratio.
//!
//! All public functions operate on `&[f64]` and return owned `Vec<f64>`
//! so the pyo3 wrapper can hand them straight back to Python without
//! extra deps (no ndarray/numpy crate). NaNs are propagated by skipping
//! the affected window.

/// Classic R/S Hurst exponent over a single window of (log) returns.
///
/// Returns NaN if the window is too small or contains non-finite values.
pub fn hurst_rs(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 16 {
        return f64::NAN;
    }
    for &v in returns {
        if !v.is_finite() {
            return f64::NAN;
        }
    }
    // Geometric ladder of sub-sample sizes.
    let mut sizes: Vec<usize> = Vec::new();
    let mut k = 8usize;
    while k <= n {
        sizes.push(k);
        k = ((k as f64) * 1.7).ceil() as usize;
    }
    if sizes.last().copied() != Some(n) {
        sizes.push(n);
    }
    let mut log_n: Vec<f64> = Vec::with_capacity(sizes.len());
    let mut log_rs: Vec<f64> = Vec::with_capacity(sizes.len());
    for &m in &sizes {
        let rs = mean_rs(returns, m);
        if rs.is_finite() && rs > 0.0 {
            log_n.push((m as f64).ln());
            log_rs.push(rs.ln());
        }
    }
    if log_n.len() < 3 {
        return f64::NAN;
    }
    linreg_slope(&log_n, &log_rs)
}

fn mean_rs(returns: &[f64], m: usize) -> f64 {
    let n = returns.len();
    if m < 4 || m > n {
        return f64::NAN;
    }
    let chunks = n / m;
    if chunks == 0 {
        return f64::NAN;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for c in 0..chunks {
        let slice = &returns[c * m..c * m + m];
        let mean = slice.iter().sum::<f64>() / (m as f64);
        let mut cum = 0.0;
        let mut cmax = f64::NEG_INFINITY;
        let mut cmin = f64::INFINITY;
        let mut sq = 0.0;
        for &x in slice {
            let d = x - mean;
            cum += d;
            if cum > cmax {
                cmax = cum;
            }
            if cum < cmin {
                cmin = cum;
            }
            sq += d * d;
        }
        let s = (sq / (m as f64)).sqrt();
        if s > 0.0 && cmax.is_finite() && cmin.is_finite() {
            sum += (cmax - cmin) / s;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / (count as f64)
    }
}

fn linreg_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let xm = x.iter().sum::<f64>() / n;
    let ym = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - xm;
        num += dx * (yi - ym);
        den += dx * dx;
    }
    if den <= 0.0 {
        f64::NAN
    } else {
        num / den
    }
}

/// Biased autocorrelation function, lags 0..=max_lag, normalized to
/// `acf[0] = 1.0`. Returns NaN-filled vec when the variance is zero.
pub fn acf(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || max_lag >= n {
        return vec![f64::NAN; max_lag + 1];
    }
    let mean = x.iter().sum::<f64>() / (n as f64);
    let mut c0 = 0.0;
    for &v in x {
        let d = v - mean;
        c0 += d * d;
    }
    c0 /= n as f64;
    if c0 <= 0.0 {
        return vec![f64::NAN; max_lag + 1];
    }
    let mut out = Vec::with_capacity(max_lag + 1);
    for lag in 0..=max_lag {
        let mut acc = 0.0;
        for i in lag..n {
            acc += (x[i] - mean) * (x[i - lag] - mean);
        }
        out.push((acc / (n as f64)) / c0);
    }
    out
}

/// Lo-MacKinlay variance ratio VR(q) on a return series.
///
/// `VR(q) = Var(q-period sums) / (q * Var(1-period))`.
/// Returns 1 under the random-walk null. < 1 = mean reversion,
/// > 1 = momentum.
pub fn variance_ratio(returns: &[f64], q: usize) -> f64 {
    let n = returns.len();
    if q < 2 || n < q * 2 {
        return f64::NAN;
    }
    let mean = returns.iter().sum::<f64>() / (n as f64);
    let mut var1 = 0.0;
    for &r in returns {
        let d = r - mean;
        var1 += d * d;
    }
    var1 /= (n - 1) as f64;
    if var1 <= 0.0 {
        return f64::NAN;
    }

    // Rolling q-period sums via running window.
    let m = n - q + 1;
    let mut sums: Vec<f64> = Vec::with_capacity(m);
    let mut window_sum: f64 = returns[0..q].iter().sum();
    sums.push(window_sum);
    for t in 1..m {
        window_sum += returns[t + q - 1] - returns[t - 1];
        sums.push(window_sum);
    }
    let sums_mean = sums.iter().sum::<f64>() / (m as f64);
    let mut var_q = 0.0;
    for &s in &sums {
        let d = s - sums_mean;
        var_q += d * d;
    }
    var_q /= (m - 1) as f64;
    let qf = q as f64;
    var_q / (qf * var1)
}

/// Rolling Hurst over a sliding window of returns; returns a Vec
/// of length `returns.len()` with NaN for `i < window-1`.
pub fn rolling_hurst(returns: &[f64], window: usize) -> Vec<f64> {
    let n = returns.len();
    let mut out = vec![f64::NAN; n];
    if window < 16 || window > n {
        return out;
    }
    for i in (window - 1)..n {
        out[i] = hurst_rs(&returns[i - window + 1..=i]);
    }
    out
}

/// Rolling lag-1 autocorrelation over a sliding window.
pub fn rolling_acf1(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    if window < 4 || window > n {
        return out;
    }
    for i in (window - 1)..n {
        let slice = &x[i - window + 1..=i];
        let a = acf(slice, 1);
        out[i] = a[1];
    }
    out
}

/// Rolling variance ratio over a sliding window.
pub fn rolling_variance_ratio(returns: &[f64], window: usize, q: usize) -> Vec<f64> {
    let n = returns.len();
    let mut out = vec![f64::NAN; n];
    if window < q * 2 || window > n {
        return out;
    }
    for i in (window - 1)..n {
        out[i] = variance_ratio(&returns[i - window + 1..=i], q);
    }
    out
}
