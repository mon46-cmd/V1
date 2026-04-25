//! Liquidity / illiquidity kernels (bar-frequency).
//!
//! - `rolling_amihud(ret, dollar_vol, window)` -- mean of |r| / dv
//!   (illiquidity; higher = more price impact per dollar).
//! - `rolling_kyle_lambda(ret, signed_vol, window)` -- OLS slope of
//!   |ret| on signed_volume; price impact per signed contract.
//! - `rolling_rolls_spread(close, window)` -- 2*sqrt(-cov(dP_t, dP_{t-1}))
//!   when the autocovariance is negative, else 0.
//!
//! All emit a Vec of the same length as the input, NaN-filled for
//! the warmup region.

fn covariance(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mx = x.iter().sum::<f64>() / nf;
    let my = y.iter().sum::<f64>() / nf;
    let mut acc = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        acc += (xi - mx) * (yi - my);
    }
    acc / (nf - 1.0)
}

/// Amihud (2002) illiquidity: rolling mean of |ret| / dollar_volume.
/// Bars where dollar_volume <= 0 are skipped from the average.
pub fn rolling_amihud(ret: &[f64], dollar_vol: &[f64], window: usize) -> Vec<f64> {
    let n = ret.len();
    let mut out = vec![f64::NAN; n];
    if dollar_vol.len() != n || window < 2 || window > n {
        return out;
    }
    for i in (window - 1)..n {
        let mut sum = 0.0;
        let mut count = 0usize;
        for j in (i + 1 - window)..=i {
            let r = ret[j];
            let dv = dollar_vol[j];
            if r.is_finite() && dv.is_finite() && dv > 0.0 {
                sum += r.abs() / dv;
                count += 1;
            }
        }
        if count > 0 {
            out[i] = sum / count as f64;
        }
    }
    out
}

/// Kyle's lambda (bar approximation): rolling OLS slope of
/// |ret| on signed_volume. Higher = more price impact per traded
/// contract. Returns NaN if signed_volume has zero variance.
pub fn rolling_kyle_lambda(ret: &[f64], signed_vol: &[f64], window: usize) -> Vec<f64> {
    let n = ret.len();
    let mut out = vec![f64::NAN; n];
    if signed_vol.len() != n || window < 4 || window > n {
        return out;
    }
    let wf = window as f64;
    for i in (window - 1)..n {
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sxx = 0.0;
        let mut sxy = 0.0;
        let mut bad = false;
        for j in (i + 1 - window)..=i {
            let x = signed_vol[j];
            let y = ret[j].abs();
            if !x.is_finite() || !y.is_finite() {
                bad = true;
                break;
            }
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        if bad {
            continue;
        }
        let den = wf * sxx - sx * sx;
        if den.abs() < 1e-18 {
            continue;
        }
        out[i] = (wf * sxy - sx * sy) / den;
    }
    out
}

/// Roll's (1984) effective spread estimator from rolling autocovariance
/// of price changes. Returns 0.0 when the autocovariance is positive
/// (estimator undefined; we surface zero rather than NaN to make
/// downstream rolling z-scoring stable).
pub fn rolling_rolls_spread(close: &[f64], window: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if window < 4 || window + 1 > n {
        return out;
    }
    // dP_t = close[t] - close[t-1]
    let mut dp: Vec<f64> = Vec::with_capacity(n);
    dp.push(f64::NAN);
    for t in 1..n {
        dp.push(close[t] - close[t - 1]);
    }
    for i in (window + 1)..n {
        let a = &dp[i + 1 - window..=i];
        let b = &dp[i - window..i];
        // Skip if any NaN sneaks in (only possible at very start).
        if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let c = covariance(a, b);
        if c < 0.0 {
            out[i] = 2.0 * (-c).sqrt();
        } else {
            out[i] = 0.0;
        }
    }
    out
}
