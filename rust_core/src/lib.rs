// Phase 6: pyo3 module exposing Tier A hot kernels.
//
// All inputs are Python lists / sequences of floats; we convert
// to Vec<f64> at the boundary and return Vec<f64> (or 3-tuples)
// which pyo3 turns back into Python lists. The Python wrappers
// in `features/layerN_*.py` are the ones that wrap into pandas.

use pyo3::prelude::*;

mod liquidity;
mod regime;
mod volume_profile;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ---- regime ----------------------------------------------------------
#[pyfunction]
fn hurst_rs(returns: Vec<f64>) -> f64 {
    regime::hurst_rs(&returns)
}

#[pyfunction]
#[pyo3(signature = (x, max_lag))]
fn acf(x: Vec<f64>, max_lag: usize) -> Vec<f64> {
    regime::acf(&x, max_lag)
}

#[pyfunction]
#[pyo3(signature = (returns, q))]
fn variance_ratio(returns: Vec<f64>, q: usize) -> f64 {
    regime::variance_ratio(&returns, q)
}

#[pyfunction]
#[pyo3(signature = (returns, window))]
fn rolling_hurst(returns: Vec<f64>, window: usize) -> Vec<f64> {
    regime::rolling_hurst(&returns, window)
}

#[pyfunction]
#[pyo3(signature = (x, window))]
fn rolling_acf1(x: Vec<f64>, window: usize) -> Vec<f64> {
    regime::rolling_acf1(&x, window)
}

#[pyfunction]
#[pyo3(signature = (returns, window, q))]
fn rolling_variance_ratio(returns: Vec<f64>, window: usize, q: usize) -> Vec<f64> {
    regime::rolling_variance_ratio(&returns, window, q)
}

// ---- liquidity -------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (ret, dollar_vol, window))]
fn rolling_amihud(ret: Vec<f64>, dollar_vol: Vec<f64>, window: usize) -> Vec<f64> {
    liquidity::rolling_amihud(&ret, &dollar_vol, window)
}

#[pyfunction]
#[pyo3(signature = (ret, signed_vol, window))]
fn rolling_kyle_lambda(ret: Vec<f64>, signed_vol: Vec<f64>, window: usize) -> Vec<f64> {
    liquidity::rolling_kyle_lambda(&ret, &signed_vol, window)
}

#[pyfunction]
#[pyo3(signature = (close, window))]
fn rolling_rolls_spread(close: Vec<f64>, window: usize) -> Vec<f64> {
    liquidity::rolling_rolls_spread(&close, window)
}

// ---- volume profile --------------------------------------------------
#[pyfunction]
#[pyo3(signature = (low, high, vol, window, nbins, area=0.70))]
fn rolling_volume_profile(
    low: Vec<f64>,
    high: Vec<f64>,
    vol: Vec<f64>,
    window: usize,
    nbins: usize,
    area: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let out = volume_profile::rolling_volume_profile(&low, &high, &vol, window, nbins, area);
    (out.poc, out.vah, out.val)
}

#[pymodule]
fn des_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(hurst_rs, m)?)?;
    m.add_function(wrap_pyfunction!(acf, m)?)?;
    m.add_function(wrap_pyfunction!(variance_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_hurst, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_acf1, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_variance_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_amihud, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_kyle_lambda, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_rolls_spread, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_volume_profile, m)?)?;
    Ok(())
}
