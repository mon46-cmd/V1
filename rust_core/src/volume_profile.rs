//! Bar-approximated rolling volume profile in Rust.
//!
//! Mirrors the pure-python `features/layer8_vp.py` algorithm:
//!  1. For each bar `i` with `i >= window-1`, take the trailing window
//!     of `(low, high, volume)`.
//!  2. Bin the price range `[min_low, max_high]` into `nbins` equal
//!     cells.
//!  3. Spread each bar's volume uniformly across the cells its
//!     `[low, high]` span overlaps (overlap / bar_span weighting).
//!  4. POC = argmax(hist); VAH/VAL = greedy expansion from POC until
//!     the value-area fraction (default 0.7) is reached, always eating
//!     the larger neighbour.
//!
//! Output: three Vec<f64> aligned to the input length, NaN for warmup
//! and for windows where the price range collapses or volume is zero.

pub struct VpOut {
    pub poc: Vec<f64>,
    pub vah: Vec<f64>,
    pub val: Vec<f64>,
}

pub fn rolling_volume_profile(
    low: &[f64],
    high: &[f64],
    vol: &[f64],
    window: usize,
    nbins: usize,
    area: f64,
) -> VpOut {
    let n = low.len();
    let mut poc = vec![f64::NAN; n];
    let mut vah = vec![f64::NAN; n];
    let mut val = vec![f64::NAN; n];
    if high.len() != n || vol.len() != n || window < 2 || window > n || nbins < 4 {
        return VpOut { poc, vah, val };
    }

    let mut hist = vec![0.0f64; nbins];
    let mut edges = vec![0.0f64; nbins + 1];
    let mut mids = vec![0.0f64; nbins];

    for i in (window - 1)..n {
        // Window range
        let mut p_lo = f64::INFINITY;
        let mut p_hi = f64::NEG_INFINITY;
        for k in (i + 1 - window)..=i {
            let l = low[k];
            let h = high[k];
            if l.is_finite() && l < p_lo {
                p_lo = l;
            }
            if h.is_finite() && h > p_hi {
                p_hi = h;
            }
        }
        if !p_lo.is_finite() || !p_hi.is_finite() || p_hi <= p_lo {
            continue;
        }
        let step = (p_hi - p_lo) / (nbins as f64);
        for b in 0..=nbins {
            edges[b] = p_lo + (b as f64) * step;
        }
        for b in 0..nbins {
            mids[b] = 0.5 * (edges[b] + edges[b + 1]);
            hist[b] = 0.0;
        }

        // Build histogram
        for k in (i + 1 - window)..=i {
            let l = low[k];
            let h = high[k];
            let v = vol[k];
            if !l.is_finite() || !h.is_finite() || !v.is_finite() || v <= 0.0 || h < l {
                continue;
            }
            // Overlap span
            let mut lo_idx = ((l - p_lo) / step).floor() as isize;
            let mut hi_idx = ((h - p_lo) / step).floor() as isize;
            if lo_idx < 0 {
                lo_idx = 0;
            }
            if hi_idx > (nbins as isize) - 1 {
                hi_idx = (nbins as isize) - 1;
            }
            if hi_idx < lo_idx {
                hi_idx = lo_idx;
            }
            let bar_span = (h - l).max(step);
            for b in lo_idx..=hi_idx {
                let bi = b as usize;
                let lo_b = edges[bi].max(l);
                let hi_b = edges[bi + 1].min(h);
                let overlap = (hi_b - lo_b).max(0.0);
                hist[bi] += v * (overlap / bar_span);
            }
        }

        // POC
        let mut total = 0.0;
        let mut poc_idx = 0usize;
        let mut peak = f64::NEG_INFINITY;
        for b in 0..nbins {
            total += hist[b];
            if hist[b] > peak {
                peak = hist[b];
                poc_idx = b;
            }
        }
        if total <= 0.0 {
            continue;
        }
        poc[i] = mids[poc_idx];
        // Value area
        let target = total * area;
        let mut lo = poc_idx;
        let mut hi = poc_idx;
        let mut accum = hist[poc_idx];
        while accum < target {
            let up_next = if hi < nbins - 1 { hist[hi + 1] } else { -1.0 };
            let dn_next = if lo > 0 { hist[lo - 1] } else { -1.0 };
            if up_next < 0.0 && dn_next < 0.0 {
                break;
            }
            if up_next >= dn_next {
                hi += 1;
                accum += hist[hi];
            } else {
                lo -= 1;
                accum += hist[lo];
            }
        }
        vah[i] = mids[hi];
        val[i] = mids[lo];
    }

    VpOut { poc, vah, val }
}
