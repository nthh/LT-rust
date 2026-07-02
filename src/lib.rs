//! LandTrendr temporal segmentation — a standalone Rust implementation.
//!
//! Per-pixel piecewise-linear segmentation of an annual spectral-index trajectory:
//!
//!   Kennedy, R.E., Yang, Z., Cohen, W.B. (2010). Detecting trends in forest
//!   disturbance and recovery using yearly Landsat time series: 1. LandTrendr —
//!   Temporal Segmentation Algorithms. Remote Sensing of Environment 114(12),
//!   2897–2910. https://doi.org/10.1016/j.rse.2010.07.008
//!
//! Validated against the native Google Earth Engine implementation:
//!   Kennedy, R.E. et al. (2018). Implementation of the LandTrendr Algorithm on
//!   Google Earth Engine. Remote Sensing 10(5), 691. https://doi.org/10.3390/rs10050691

// LandTrendr temporal segmentation
// ---------------------------------------------------------------------------

/// LandTrendr algorithm parameters.
pub struct LandTrendrParams {
    pub max_segments: usize,
    pub spike_threshold: f32,
    pub vertex_count_overshoot: usize,
    pub recovery_threshold: f32,
    pub p_value_threshold: f64,
    pub best_model_proportion: f64,
    pub min_observations_needed: usize,
    pub prevent_one_year_recovery: bool,
}

impl Default for LandTrendrParams {
    fn default() -> Self {
        Self {
            max_segments: 6,
            spike_threshold: 0.9,
            vertex_count_overshoot: 3,
            recovery_threshold: 0.25,
            p_value_threshold: 0.05,
            best_model_proportion: 0.75,
            min_observations_needed: 6,
            prevent_one_year_recovery: true,
        }
    }
}

/// Result of LandTrendr on a single pixel.
pub struct LandTrendrPixelResult {
    pub fitted: Vec<f32>,
    pub is_vertex: Vec<bool>,
    pub rmse: f32,
    pub segments: Vec<SegmentInfo>,
}

pub struct SegmentInfo {
    pub start_year: i32,
    pub end_year: i32,
    pub start_val: f32,
    pub end_val: f32,
    pub magnitude: f32,
    pub duration: i32,
    pub rate: f32,
}

/// Run LandTrendr segmentation on a single pixel time series.
pub fn pixel(
    values: &[f32],
    years: &[i32],
    params: &LandTrendrParams,
) -> LandTrendrPixelResult {
    let n = values.len();
    assert_eq!(n, years.len(), "values and years must have same length");
    assert!(n <= LT_MAX_N, "time series too long for workspace (max {})", LT_MAX_N);

    let n_valid = values.iter().filter(|v| !v.is_nan()).count();
    if n_valid < params.min_observations_needed {
        return LandTrendrPixelResult {
            fitted: values.to_vec(),
            is_vertex: vec![false; n],
            rmse: 0.0,
            segments: Vec::new(),
        };
    }

    // Delegate to the fast workspace-based implementation (single algorithm path)
    let mut ws = LandTrendrWorkspace::new();
    let selected = pixel_core(values, years, n, params, &mut ws);

    // Extract full results from workspace
    let fitted = ws.fitted[..n].to_vec();
    let nv = ws.cand_n_verts[selected];
    let final_verts: Vec<usize> = ws.cand_verts[selected][..nv].to_vec();

    let mut is_vertex = vec![false; n];
    for &vi in &final_verts {
        is_vertex[vi] = true;
    }

    let mut sum_sq: f64 = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        if !values[i].is_nan() {
            let d = (values[i] - fitted[i]) as f64;
            sum_sq += d * d;
            count += 1;
        }
    }
    let rmse = if count > 0 { (sum_sq / count as f64).sqrt() as f32 } else { 0.0 };

    let segments = extract_segments(years, &fitted, &final_verts);

    LandTrendrPixelResult { fitted, is_vertex, rmse, segments }
}

/// Debug tape for differential validation against the LT-IDL reference. Returns
/// (despiked series, stage-② candidate vertex indices, stage-③ vetted vertex
/// indices), mirroring the front half of `pixel_core` exactly so
/// the intermediate vertex sets can be diffed stage-by-stage against IDL.
pub fn pixel_debug(
    values: &[f32], years: &[i32], params: &LandTrendrParams,
) -> (Vec<f32>, Vec<usize>, Vec<usize>) {
    let n = values.len();
    let mut ws = LandTrendrWorkspace::new();
    let mut despiked = [0.0f32; LT_MAX_N];
    interpolate_nans_into(values, years, n, &mut despiked);
    despike_inplace(&mut despiked, n, params.spike_threshold);
    let (year_range, val_range) = compute_ranges(&despiked, n, years);
    let nv_c = detect_vertices(
        &despiked, years, n, params.max_segments, params.vertex_count_overshoot, &mut ws,
    );
    let candidates = ws.vertices[..nv_c].to_vec();
    let nv_v = cull_vertices(
        &despiked, years, nv_c, params.max_segments, year_range, val_range, &mut ws,
    );
    let vetted = ws.vertices[..nv_v].to_vec();
    (despiked[..n].to_vec(), candidates, vetted)
}

// ---------------------------------------------------------------------------
// Shared helpers (used by both pixel and flat paths)
// ---------------------------------------------------------------------------

fn extract_segments(years: &[i32], fitted: &[f32], vertex_indices: &[usize]) -> Vec<SegmentInfo> {
    let mut verts = vertex_indices.to_vec();
    verts.sort_unstable();
    let mut segments = Vec::new();

    for i in 0..verts.len().saturating_sub(1) {
        let i_start = verts[i];
        let i_end = verts[i + 1];
        let start_year = years[i_start];
        let end_year = years[i_end];
        let start_val = fitted[i_start];
        let end_val = fitted[i_end];
        let magnitude = end_val - start_val;
        let duration = end_year - start_year;
        let rate = if duration > 0 {
            magnitude / duration as f32
        } else {
            0.0
        };

        segments.push(SegmentInfo {
            start_year,
            end_year,
            start_val,
            end_val,
            magnitude,
            duration,
            rate,
        });
    }

    segments
}

/// Survival function (upper tail) of the F-distribution: P(F > f_stat) for
/// (df1, df2) degrees of freedom. Exact, via the regularized incomplete beta —
///   P(F <= f) = I_{d1 f/(d1 f + d2)}(d1/2, d2/2),  so the survival is
///   1 - that = I_{d2/(d1 f + d2)}(d2/2, d1/2).
/// This is exactly LT-IDL's `1 - f_test1(f, df1, df2)` (f_test1 being the
/// incomplete-beta F CDF), replacing the earlier Wilson-Hilferty approximation,
/// which tipped borderline model-selection p-values across the pval threshold.
fn f_survival(f_stat: f64, df1: f64, df2: f64) -> f64 {
    if f_stat <= 0.0 {
        return 1.0;
    }
    let x = df2 / (df1 * f_stat + df2);
    betai(df2 / 2.0, df1 / 2.0, x)
}

/// Regularized incomplete beta function I_x(a, b) (Numerical Recipes `betai`).
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let front = (a * x.ln() + b * (1.0 - x).ln() + ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for the incomplete beta (Lentz's method).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 200;
    const EPS: f64 = 3.0e-12;
    const FPMIN: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Natural log of the gamma function (Lanczos approximation, g=7).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COF: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection: Γ(x)Γ(1-x) = π / sin(πx)
        std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().abs().ln()
            - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COF[0];
        let t = x + G + 0.5;
        for (i, &c) in COF.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

// ---------------------------------------------------------------------------
// Zero-allocation per-pixel workspace
// ---------------------------------------------------------------------------
//
// The per-pixel core runs in a hot loop over up to ~240K raster pixels. A naive
// implementation allocates dozens of Vecs per call (per-segment arrays, fit
// buffers), which dominates wall time under simple allocators (e.g. WASM's
// dlmalloc). The workspace pre-allocates every buffer once and reuses it across
// pixels, holding heap allocations in the hot loop to zero.

const LT_MAX_N: usize = 128;
const LT_MAX_VERTS: usize = 24;
const LT_MAX_CANDIDATES: usize = 12;

struct LandTrendrWorkspace {
    vertices: [usize; LT_MAX_VERTS],
    work_verts: [usize; LT_MAX_VERTS],
    cand_verts: [[usize; LT_MAX_VERTS]; LT_MAX_CANDIDATES],
    cand_n_verts: [usize; LT_MAX_CANDIDATES],
    cand_ssr: [f64; LT_MAX_CANDIDATES],
    fitted: [f32; LT_MAX_N],
}

impl LandTrendrWorkspace {
    fn new() -> Self {
        Self {
            vertices: [0; LT_MAX_VERTS],
            work_verts: [0; LT_MAX_VERTS],
            cand_verts: [[0; LT_MAX_VERTS]; LT_MAX_CANDIDATES],
            cand_n_verts: [0; LT_MAX_CANDIDATES],
            cand_ssr: [0.0; LT_MAX_CANDIDATES],
            fitted: [0.0; LT_MAX_N],
        }
    }
}

/// Least-squares line fit returning (slope, intercept). Zero allocation.
#[inline]
fn fit_line_coeffs(values: &[f32], years: &[i32], start: usize, end: usize) -> (f64, f64) {
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut count = 0usize;
    for i in start..=end {
        let y = values[i] as f64;
        if !y.is_nan() {
            let x = years[i] as f64;
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            count += 1;
        }
    }
    if count < 2 {
        let mean = if count > 0 { sum_y / count as f64 } else { 0.0 };
        return (0.0, mean);
    }
    let denom = count as f64 * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return (0.0, sum_y / count as f64);
    }
    let slope = (count as f64 * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / count as f64;
    (slope, intercept)
}

/// Vertex importance, faithful to LT-IDL `angle_diff` (segmentation/angle_diff.pro)
/// as used by `vet_verts3`: the steeper adjoining segment's slope-angle, multiplied
/// by a disturbance weight (distweightfactor=2). Computed in IDL's square-aspect
/// coordinate (y scaled to the data range, then to the year span) and in the loss-UP
/// orientation (IDL applies modifier=-1 before segmentation), so a vertex whose
/// following segment is a loss is up-weighted. Larger = more important; the cull
/// removes the MINIMUM. `year_range`/`val_range` are the full-series spans.
#[inline]
fn vertex_importance(
    values: &[f32], years: &[i32],
    i_left: usize, i_center: usize, i_right: usize,
    year_range: f64, val_range: f64,
) -> f64 {
    const DISTWEIGHTFACTOR: f64 = 2.0;
    // Square-aspect, loss-up scaled y-differences: dy = -(Δvalue)/val_range * year_range
    // (negated because LT-rs carries the loss-down series; IDL segments loss-up).
    let dy1 = -((values[i_center] - values[i_left]) as f64) / val_range * year_range;
    let dy2 = -((values[i_right] - values[i_center]) as f64) / val_range * year_range;
    let dx1 = (years[i_center] - years[i_left]) as f64;
    let dx2 = (years[i_right] - years[i_center]) as f64;
    let angle1 = if dx1 != 0.0 { (dy1 / dx1).atan() } else { 0.0 };
    let angle2 = if dx2 != 0.0 { (dy2 / dx2).atan() } else { 0.0 };
    // Disturbance boost (angle_diff.pro:67): reward a vertex whose FOLLOWING segment
    // is a loss (dy2 > 0 in the loss-up view). IDL's yrange = range(yscale) = year_range.
    let scaler = (DISTWEIGHTFACTOR * dy2 / year_range).max(0.0) + 1.0;
    angle1.abs().max(angle2.abs()) * scaler
}

#[inline]
fn compute_ranges(values: &[f32], n: usize, years: &[i32]) -> (f64, f64) {
    let year_range = (years[n - 1] - years[0]).max(1) as f64;
    let mut val_min = f64::INFINITY;
    let mut val_max = f64::NEG_INFINITY;
    for i in 0..n {
        let v = values[i] as f64;
        if !v.is_nan() {
            if v < val_min { val_min = v; }
            if v > val_max { val_max = v; }
        }
    }
    let val_range = if (val_max - val_min).abs() < 1e-15 { 1.0 } else { val_max - val_min };
    (year_range, val_range)
}

#[inline]
fn interpolate_nans_into(values: &[f32], years: &[i32], n: usize, out: &mut [f32]) {
    out[..n].copy_from_slice(&values[..n]);
    let mut valid_years = [0.0f64; LT_MAX_N];
    let mut valid_vals = [0.0f64; LT_MAX_N];
    let mut n_valid = 0usize;
    for i in 0..n {
        if !values[i].is_nan() {
            valid_years[n_valid] = years[i] as f64;
            valid_vals[n_valid] = values[i] as f64;
            n_valid += 1;
        }
    }
    if n_valid == 0 || n_valid == n { return; }
    let xs = &valid_years[..n_valid];
    let ys = &valid_vals[..n_valid];
    for i in 0..n {
        if out[i].is_nan() {
            let x = years[i] as f64;
            out[i] = if x <= xs[0] {
                ys[0] as f32
            } else if x >= xs[n_valid - 1] {
                ys[n_valid - 1] as f32
            } else {
                let mut val = ys[n_valid - 1];
                for j in 0..n_valid - 1 {
                    if x >= xs[j] && x <= xs[j + 1] {
                        let t = (x - xs[j]) / (xs[j + 1] - xs[j]);
                        val = ys[j] + t * (ys[j + 1] - ys[j]);
                        break;
                    }
                }
                val as f32
            };
        }
    }
}

/// Despike — LT-IDL `desawtooth` (segmentation/desawtooth.pro). Each iteration
/// corrects the single spikiest point by a PARTIAL pull toward its neighbor
/// midpoint, then recomputes. IDL's `while prop gt stopat` tests the PREVIOUS
/// iteration's peak, so it applies one correction PAST the threshold: the point
/// whose peak first dips below stopat is still corrected. Endpoints are fixed
/// (prop 0), so a spike-free series gets a single no-op pass.
///   prop[i] = 1 - |v[i-1]-v[i+1]| / max(|v[i]-v[i-1]|, |v[i]-v[i+1]|)
///   v[i]   += prop[i] * (midpoint(neighbors) - v[i])
/// (The earlier single-pass full-midpoint replacement — and a check-then-correct
/// loop — under-corrected, leaving marginal spikes that tipped model selection on
/// noisy pixels; see idl-harness/ idl_arid_rootcause.py.)
#[inline]
fn despike_inplace(values: &mut [f32], n: usize, spike_threshold: f32) {
    if spike_threshold >= 1.0 || n < 3 { return; }
    let stopat = spike_threshold as f64;
    let mut prop = 1.0f64; // IDL sentinel: forces the first iteration
    let max_iters = 4 * n; // safety bound
    for _ in 0..max_iters {
        if prop <= stopat { break; } // IDL `while prop gt stopat` — uses the PREVIOUS peak
        let mut best_prop = 0.0f64; // endpoints contribute prop 0 / correction 0
        let mut best_i = 0usize;
        let mut best_corr = 0.0f64;
        for i in 1..n - 1 {
            let vi = values[i] as f64;
            let vl = values[i - 1] as f64;
            let vr = values[i + 1] as f64;
            let mut md = (vi - vl).abs().max((vi - vr).abs());
            let diff_2 = (vl - vr).abs(); // |v[i-1]-v[i+1]|
            if md == 0.0 { md = diff_2; }
            if md == 0.0 { continue; }
            let prop_i = 1.0 - diff_2 / md;
            if prop_i > best_prop {
                best_prop = prop_i;
                best_i = i;
                best_corr = prop_i * ((vl + vr) / 2.0 - vi);
            }
        }
        prop = best_prop;
        values[best_i] = (values[best_i] as f64 + best_corr) as f32; // correct (no-op if endpoint)
    }
}

/// Simultaneous least-squares piecewise-linear fit. Zero allocation (stack arrays).
///
/// Sequential anchored fit — LT-IDL `find_best_trace` + `anchored_regression`
/// (segmentation/tbcd_v2.pro:466, util/helper/anchored_regression.pro), the IDL
/// PRIMARY fit. Fits segments left→right with floating vertex values: segment 0
/// chooses free OLS vs the point-to-point line by lower SSE; each later segment
/// chooses a regression ANCHORED at the prior segment's fitted end value vs
/// point-to-point. Sharper, data-faithful breakpoints than a simultaneous joint
/// solve (the IDL float-all fallback `find_best_trace3`), which over/undershoots
/// shared vertices at sharp V's. `values` are gap-filled.
#[inline]
fn fit_segments_anchored(
    values: &[f32], years: &[i32],
    verts: &[usize], n_verts: usize,
    n: usize, fitted_out: &mut [f32],
) {
    if n_verts < 2 {
        fitted_out[..n].copy_from_slice(&values[..n]);
        return;
    }
    let mut vv = [0.0f32; LT_MAX_VERTS];
    for i in 0..n_verts {
        vv[i] = values[verts[i]];
    }

    for s in 0..n_verts - 1 {
        let a = verts[s];
        let b = verts[s + 1];
        let xa = years[a] as f32;
        let span = years[b] as f32 - xa;

        // Alternative to point-to-point: free OLS for segment 0, else a regression
        // anchored at vv[s] (the prior segment's fitted end value).
        let (alt_start, alt_slope) = if s == 0 {
            let (mut sx, mut sy, mut sxx, mut sxy, mut cnt) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
            for j in a..=b {
                let xj = (years[j] as f32 - xa) as f64;
                let yj = values[j] as f64;
                sx += xj; sy += yj; sxx += xj * xj; sxy += xj * yj; cnt += 1.0;
            }
            let denom = cnt * sxx - sx * sx;
            let slope = if denom != 0.0 { (cnt * sxy - sx * sy) / denom } else { 0.0 };
            (((sy - slope * sx) / cnt) as f32, slope as f32)
        } else {
            let anchor = vv[s] as f64;
            let (mut sxx, mut sxg) = (0.0f64, 0.0f64);
            for j in a..=b {
                let xj = (years[j] as f32 - xa) as f64;
                sxx += xj * xj;
                sxg += xj * (values[j] as f64 - anchor);
            }
            (vv[s], (if sxx != 0.0 { sxg / sxx } else { 0.0 }) as f32)
        };

        // pick_better_fit: point-to-point (vv[s] -> vv[s+1]) vs the alternative,
        // by lower SSE (ties favor point-to-point).
        let (dot_start, dot_end) = (vv[s], vv[s + 1]);
        let (mut sse_dot, mut sse_alt) = (0.0f64, 0.0f64);
        for j in a..=b {
            let dx = years[j] as f32 - xa;
            let t = if span != 0.0 { dx / span } else { 0.0 };
            let dotv = dot_start + t * (dot_end - dot_start);
            let altv = alt_start + dx * alt_slope;
            sse_dot += ((values[j] - dotv) as f64).powi(2);
            sse_alt += ((values[j] - altv) as f64).powi(2);
        }
        let use_alt = sse_alt < sse_dot;

        for j in a..=b {
            let dx = years[j] as f32 - xa;
            fitted_out[j] = if use_alt {
                alt_start + dx * alt_slope
            } else {
                let t = if span != 0.0 { dx / span } else { 0.0 };
                dot_start + t * (dot_end - dot_start)
            };
        }
        vv[s] = fitted_out[a];
        vv[s + 1] = fitted_out[b];
    }
}

/// Identify vertices using iterative max-residual. Zero allocation.
fn detect_vertices(
    values: &[f32], years: &[i32], n: usize,
    max_segments: usize, overshoot: usize,
    ws: &mut LandTrendrWorkspace,
) -> usize {
    let target = (max_segments + 1 + overshoot).min(n);
    ws.vertices[0] = 0;
    ws.vertices[1] = n - 1;
    let mut nv = 2usize;

    while nv < target {
        let mut best_residual = -1.0f64;
        let mut best_idx: Option<usize> = None;
        for s in 0..nv - 1 {
            let seg_start = ws.vertices[s];
            let seg_end = ws.vertices[s + 1];
            if seg_end - seg_start <= 1 { continue; }
            let (slope, intercept) = fit_line_coeffs(values, years, seg_start, seg_end);
            for i in (seg_start + 1)..seg_end {
                let fitted = intercept + slope * years[i] as f64;
                let residual = (values[i] as f64 - fitted).abs();
                if residual > best_residual {
                    best_residual = residual;
                    best_idx = Some(i);
                }
            }
        }
        match best_idx {
            Some(idx) if best_residual > 0.0 => {
                let mut exists = false;
                for i in 0..nv {
                    if ws.vertices[i] == idx { exists = true; break; }
                }
                if exists { break; }
                // Insert in sorted position
                let mut pos = nv;
                for i in 0..nv {
                    if ws.vertices[i] > idx { pos = i; break; }
                }
                let mut j = nv;
                while j > pos { ws.vertices[j] = ws.vertices[j - 1]; j -= 1; }
                ws.vertices[pos] = idx;
                nv += 1;
            }
            _ => break,
        }
    }

    nv
}

/// Stage ③: cull excess candidate vertices to max_segments+1 by importance,
/// faithful to LT-IDL `vet_verts3`: each round, remove the interior vertex with the
/// MINIMUM `vertex_importance` (the flattest, least disturbance-relevant), then
/// recompute. This protects steep disturbance/recovery vertices that a pure interior
/// angle would discard — see idl-harness/ tape validation (idl_tape_diff.py).
fn cull_vertices(
    values: &[f32], years: &[i32],
    mut nv: usize, max_segments: usize,
    year_range: f64, val_range: f64,
    ws: &mut LandTrendrWorkspace,
) -> usize {
    let target_count = max_segments + 1;
    while nv > target_count {
        let mut min_importance = f64::INFINITY;
        let mut min_idx: Option<usize> = None;
        for i in 1..nv - 1 {
            let imp = vertex_importance(
                values, years,
                ws.vertices[i - 1], ws.vertices[i], ws.vertices[i + 1],
                year_range, val_range,
            );
            if imp < min_importance { min_importance = imp; min_idx = Some(i); }
        }
        match min_idx {
            Some(idx) => {
                for i in idx..nv - 1 { ws.vertices[i] = ws.vertices[i + 1]; }
                nv -= 1;
            }
            None => break,
        }
    }
    nv
}

fn identify_vertices(
    values: &[f32], years: &[i32], n: usize,
    max_segments: usize, overshoot: usize,
    year_range: f64, val_range: f64,
    ws: &mut LandTrendrWorkspace,
) -> usize {
    let nv = detect_vertices(values, years, n, max_segments, overshoot, ws);
    cull_vertices(values, years, nv, max_segments, year_range, val_range, ws)
}

/// Build candidate models and select best via F-test. Zero allocation.
/// ws.fitted contains the selected model's fitted values on return.
fn fit_and_select(
    values: &[f32], years: &[i32], n: usize,
    n_verts: usize, params: &LandTrendrParams,
    year_range: f64, val_range: f64,
    ws: &mut LandTrendrWorkspace,
) -> usize {
    let n_valid = values[..n].iter().filter(|v| !v.is_nan()).count();

    // Working copy of the despiked series. LT-IDL `take_out_weakest2` interpolates a
    // removed recovery-violator's data point IN PLACE (tbcd_v2.pro:657-667), so later
    // models in the ladder are fit against the flattened series. Without it, rust's
    // anchored fit chases the removed peak and overshoots the next vertex.
    let mut work_vals = [0.0f32; LT_MAX_N];
    work_vals[..n].copy_from_slice(&values[..n]);

    ws.work_verts[..n_verts].copy_from_slice(&ws.vertices[..n_verts]);
    let mut n_wv = n_verts;
    let mut n_cand = 0usize;

    while n_wv >= 2 && n_cand < LT_MAX_CANDIDATES {
        ws.cand_verts[n_cand][..n_wv].copy_from_slice(&ws.work_verts[..n_wv]);
        ws.cand_n_verts[n_cand] = n_wv;

        fit_segments_anchored(&work_vals, years, &ws.work_verts, n_wv, n, &mut ws.fitted);
        let mut ssr = 0.0f64;
        for i in 0..n {
            if !work_vals[i].is_nan() {
                let d = (work_vals[i] - ws.fitted[i]) as f64;
                ssr += d * d;
            }
        }
        ws.cand_ssr[n_cand] = ssr;
        n_cand += 1;

        if n_wv <= 2 { break; }

        // Build the next simpler candidate the way LT-IDL `take_out_weakest2`
        // (tbcd_v2.pro:608) does — this is the model-ladder drop ORDER:
        //   (1) if a recovery segment is too steep (|slope|/range > recovery_threshold;
        //       loss-down => recovery is a POSITIVE slope), drop the LATTER (interior)
        //       vertex of the steepest such segment;
        //   (2) otherwise drop the interior vertex with the least LOCAL triangle-MSE
        //       (bridge its two neighbors with a straight line) — a cheap local
        //       penalty, NOT a global refit.
        // ws.fitted holds the current model's fit, so ws.fitted[vertex] are the vertvals.
        let mut vmin = f64::INFINITY;
        let mut vmax = f64::NEG_INFINITY;
        for i in 0..n {
            let v = ws.fitted[i] as f64;
            if v < vmin { vmin = v; }
            if v > vmax { vmax = v; }
        }
        let vrange = (vmax - vmin).max(1e-9);

        let mut remove: Option<usize> = None;
        let mut from_recovery = false;
        let mut worst_scaled = params.recovery_threshold as f64;
        for s in 0..n_wv - 1 {
            let latter = s + 1;
            if latter > n_wv - 2 { continue; } // never drop the last vertex (IDL interpolates instead)
            let a = ws.work_verts[s];
            let b = ws.work_verts[latter];
            let dx = (years[b] - years[a]) as f64;
            if dx <= 0.0 { continue; }
            let slope = (ws.fitted[b] - ws.fitted[a]) as f64 / dx;
            if slope > 0.0 {
                let scaled = slope / vrange; // recovery steepness (loss-down: recovery is +slope)
                if scaled > worst_scaled {
                    worst_scaled = scaled;
                    remove = Some(latter);
                    from_recovery = true;
                }
            }
        }

        if remove.is_none() {
            let mut best_mse = f64::INFINITY;
            for i in 1..n_wv - 1 {
                let a = ws.work_verts[i - 1];
                let c = ws.work_verts[i + 1];
                let (va, vc) = (ws.fitted[a] as f64, ws.fitted[c] as f64);
                let (xa, xc) = (years[a] as f64, years[c] as f64);
                let span = (xc - xa).max(1e-9);
                let mut sse = 0.0f64;
                for j in a..=c {
                    if work_vals[j].is_nan() { continue; }
                    let t = (years[j] as f64 - xa) / span;
                    let line = va + t * (vc - va);
                    let d = work_vals[j] as f64 - line;
                    sse += d * d;
                }
                let mse = sse / span;
                if mse < best_mse { best_mse = mse; remove = Some(i); }
            }
        }

        match remove {
            Some(idx) => {
                // IDL interpolates the removed recovery-violator's data point in place,
                // between its immediate time-series neighbors, so later fits don't chase
                // the removed peak (tbcd_v2.pro:657-667).
                let rv = ws.work_verts[idx];
                if from_recovery && rv > 0 && rv + 1 < n {
                    let (lx, rx) = (years[rv - 1] as f64, years[rv + 1] as f64);
                    let (lv, rvv) = (work_vals[rv - 1] as f64, work_vals[rv + 1] as f64);
                    let slope = if rx != lx { (rvv - lv) / (rx - lx) } else { 0.0 };
                    work_vals[rv] = (lv + (years[rv] as f64 - lx) * slope) as f32;
                }
                for i in idx..n_wv - 1 { ws.work_verts[i] = ws.work_verts[i + 1]; }
                n_wv -= 1;
            }
            None => break,
        }
    }

    if n_cand == 0 {
        let (slope, intercept) = fit_line_coeffs(&work_vals, years, 0, n - 1);
        for i in 0..n { ws.fitted[i] = (intercept + slope * years[i] as f64) as f32; }
        ws.cand_verts[0][0] = 0;
        ws.cand_verts[0][1] = n - 1;
        ws.cand_n_verts[0] = 2;
        return 0;
    }

    let full_ssr = ws.cand_ssr[0];
    if full_ssr < 1e-10 {
        fit_segments_anchored(
            &work_vals, years, &ws.cand_verts[0], ws.cand_n_verts[0], n, &mut ws.fitted,
        );
        return 0;
    }

    // Model selection — LT-IDL tbcd_v2 `pick_best_model6` + the flat-line rule.
    // Score each candidate by its p-of-F vs a FLAT line (calc_fitting_stats3):
    //   ss_regr = ss_total - ss_resid,  n_predictors = 2*(V-1),
    //   f = (ss_regr/df_regr)/(ss_resid/df_resid),  p_of_f = 1 - F_cdf(f).
    // Take the most-complex model within (2 - bmp) x the best p-of-F, then collapse
    // to a flat line at mean(y) if even that model is not significant (p_of_f >
    // pval). The flat-line rule is what suppresses cyclic (e.g. cropland) noise that
    // the previous always-keep-full selection turned into false disturbances.
    let mut mean_y = 0.0f64;
    for i in 0..n {
        if !work_vals[i].is_nan() { mean_y += work_vals[i] as f64; }
    }
    mean_y /= n_valid.max(1) as f64;
    let mut ss_total = 0.0f64;
    for i in 0..n {
        if !work_vals[i].is_nan() {
            let d = work_vals[i] as f64 - mean_y;
            ss_total += d * d;
        }
    }

    let mut p_of_f = [1.0f64; LT_MAX_CANDIDATES];
    for idx in 0..n_cand {
        let n_pred = 2 * ws.cand_n_verts[idx].saturating_sub(1); // IDL ((n_vertices)*2)-2
        let df_regr = n_pred as f64;
        let df_resid = n_valid as f64 - n_pred as f64 - 1.0;
        if df_regr <= 0.0 || df_resid <= 0.0 { continue; } // p_of_f stays 1.0
        let mut ss_resid = ws.cand_ssr[idx];
        if ss_resid > ss_total { ss_resid = ss_total; } // IDL clamp for rounding
        let ms_regr = (ss_total - ss_resid) / df_regr;
        let ms_resid = ss_resid / df_resid;
        let f_regr = if ms_regr < 1e-5 || ms_resid <= 0.0 { 1e-5 } else { ms_regr / ms_resid };
        p_of_f[idx] = f_survival(f_regr, df_regr, df_resid).clamp(0.0, 1.0);
    }

    // pick_best_model6 (use_fstat=0): most-complex model (lowest index = most
    // vertices) within (2 - best_model_proportion) x the minimum p-of-F.
    let min_pof = p_of_f[..n_cand].iter().cloned().fold(f64::INFINITY, f64::min);
    let thresh = (2.0 - params.best_model_proportion) * min_pof;
    let mut selected = 0usize;
    for idx in 0..n_cand {
        if p_of_f[idx] <= thresh { selected = idx; break; }
    }

    // Flat-line rule (tbcd_v2:1430): no significant model -> horizontal line at the
    // mean, reported as a single flat segment (i.e. no disturbance).
    if p_of_f[selected] > params.p_value_threshold {
        for i in 0..n { ws.fitted[i] = mean_y as f32; }
        ws.cand_verts[selected][0] = 0;
        ws.cand_verts[selected][1] = n - 1;
        ws.cand_n_verts[selected] = 2;
        return selected;
    }

    fit_segments_anchored(
        &work_vals, years,
        &ws.cand_verts[selected], ws.cand_n_verts[selected], n, &mut ws.fitted,
    );

    selected
}

/// Core LandTrendr implementation — single algorithm path.
/// Runs despike → vertex identification → model selection → recovery clamp.
/// Returns selected candidate index; results are in ws.fitted and ws.cand_verts.
#[inline]
fn pixel_core(
    values: &[f32], years: &[i32], n: usize,
    params: &LandTrendrParams, ws: &mut LandTrendrWorkspace,
) -> usize {
    let mut despiked = [0.0f32; LT_MAX_N];
    interpolate_nans_into(values, years, n, &mut despiked);
    despike_inplace(&mut despiked, n, params.spike_threshold);

    let (year_range, val_range) = compute_ranges(&despiked, n, years);

    let mut n_verts = identify_vertices(
        &despiked, years, n,
        params.max_segments, params.vertex_count_overshoot,
        year_range, val_range, ws,
    );

    // prevent_one_year_recovery: a disturbance bottom immediately followed by a
    // single-year recovery is almost always residual cloud/shadow rather than real
    // regrowth (eMapR LT-GEE runParam). Drop the vertex that ENDS such a 1-year
    // recovery so the recovery is forced to span >=2 years (loss-down NBR: a drop
    // into v[i] then a rise out of v[i] over one year). Done on the vertex set
    // before model fitting, so every candidate inherits the constraint.
    if params.prevent_one_year_recovery {
        let mut i = 1;
        while i + 1 < n_verts {
            let (a, b, c) = (ws.vertices[i - 1], ws.vertices[i], ws.vertices[i + 1]);
            let drop_in = despiked[b] < despiked[a];          // disturbance into the bottom
            let rise_out = despiked[c] > despiked[b];          // recovery out of the bottom
            let one_year = years[c] - years[b] == 1;
            if drop_in && rise_out && one_year {
                for k in (i + 1)..n_verts - 1 { ws.vertices[k] = ws.vertices[k + 1]; }
                n_verts -= 1;                                  // removed the 1-yr recovery endpoint
            } else {
                i += 1;
            }
        }
    }

    let selected = fit_and_select(
        &despiked, years, n,
        n_verts, params, year_range, val_range, ws,
    );

    // Recovery clamp: after fitting, constrain recovery segment slopes.
    // Clamp vertex endpoints so rate <= recovery_threshold, re-interpolate.
    if params.recovery_threshold < 1.0 {
        let nv = ws.cand_n_verts[selected];
        let verts = &ws.cand_verts[selected][..nv];
        let mut changed = false;
        for i in 0..nv.saturating_sub(1) {
            let si = verts[i];
            let ei = verts[i + 1];
            if ei < n && si < n {
                let mag = ws.fitted[ei] - ws.fitted[si];
                let dur = (years[ei] - years[si]) as f32;
                if mag > 0.0 && dur > 0.0 && mag / dur > params.recovery_threshold {
                    ws.fitted[ei] = ws.fitted[si] + params.recovery_threshold * dur;
                    changed = true;
                }
            }
        }
        if changed {
            for i in 0..nv.saturating_sub(1) {
                let si = verts[i];
                let ei = verts[i + 1];
                if ei < n && si < n {
                    let sv = ws.fitted[si];
                    let ev = ws.fitted[ei];
                    let span = (years[ei] - years[si]) as f32;
                    if span > 0.0 {
                        for j in (si + 1)..ei {
                            ws.fitted[j] = sv + (years[j] - years[si]) as f32 / span * (ev - sv);
                        }
                    }
                }
            }
        }
    }

    selected
}

/// Fast per-pixel LandTrendr.
/// Returns (net_magnitude, disturbance_year, rmse, peak_to_trough_magnitude).
///
/// - `net_magnitude` = fitted[last] - fitted[first] (net change; back-compat band 0).
/// - `peak_to_trough_magnitude` = fitted[trough_idx] - fitted[peak_idx] over the
///   FULL fitted trajectory, where peak_idx = argmax(fitted), trough_idx =
///   argmin(fitted); set to 0.0 when trough_idx <= peak_idx (monotonic rise / no
///   disturbance). This is the standard LandTrendr disturbance-depth statistic
///   and matches the validated Python path (extract.py in the Bootleg-MTBS run):
///       peak_idx = argmax(fitted); trough_idx = argmin(fitted)
///       magnitude = fitted[trough_idx] - fitted[peak_idx]   (<= 0)
///       magnitude[trough_idx <= peak_idx] = 0.0
///   Returns NaN for both magnitudes when the pixel has insufficient valid
///   observations (so callers can mask on isfinite, matching extract.py's
///   `magnitude[~valid] = NaN`).
#[inline]
fn pixel_summary(
    values: &[f32], years: &[i32], n: usize,
    params: &LandTrendrParams, ws: &mut LandTrendrWorkspace,
) -> (f32, f32, f32, f32) {
    let n_valid = values[..n].iter().filter(|v| !v.is_nan()).count();
    if n_valid < params.min_observations_needed {
        // Insufficient data: net_change keeps its historical 0.0 sentinel for
        // band-0 back-compat; peak-to-trough is NaN so it masks out as invalid.
        return (0.0, f32::NAN, 0.0, f32::NAN);
    }

    let selected = pixel_core(values, years, n, params, ws);

    // RMSE
    let mut sum_sq: f64 = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        if !values[i].is_nan() {
            let d = (values[i] - ws.fitted[i]) as f64;
            sum_sq += d * d;
            count += 1;
        }
    }
    let rmse = if count > 0 { (sum_sq / count as f64).sqrt() as f32 } else { 0.0 };

    let net_change = ws.fitted[n - 1] - ws.fitted[0];

    let verts = &ws.cand_verts[selected];
    let nv = ws.cand_n_verts[selected];
    let mut max_mag = 0.0f32;
    let mut dist_year = f32::NAN;
    for i in 0..nv.saturating_sub(1) {
        let magnitude = ws.fitted[verts[i + 1]] - ws.fitted[verts[i]];
        if magnitude < max_mag {
            max_mag = magnitude;
            dist_year = years[verts[i]] as f32;
        }
    }

    // Peak-to-trough over the full fitted trajectory (standard disturbance
    // depth). argmax/argmin scan; tie-break = first index (matches numpy argmax/
    // argmin, which return the first occurrence). The fitted trajectory has no
    // NaNs over [0, n) for a fitted pixel, so no NaN guard is needed here.
    let mut peak_idx = 0usize;
    let mut trough_idx = 0usize;
    let mut peak_val = ws.fitted[0];
    let mut trough_val = ws.fitted[0];
    for i in 1..n {
        let v = ws.fitted[i];
        if v > peak_val {
            peak_val = v;
            peak_idx = i;
        }
        if v < trough_val {
            trough_val = v;
            trough_idx = i;
        }
    }
    let peak_to_trough = if trough_idx <= peak_idx {
        0.0 // monotonic rise (or trough precedes peak) => no disturbance
    } else {
        trough_val - peak_val // <= 0
    };

    (net_change, dist_year, rmse, peak_to_trough)
}

/// Run LandTrendr on a full raster stack.
///
/// `data`: flat slice (band_count bands of pixel_count pixels each)
/// `pixel_count`: pixels per band
/// `band_count`: number of annual observations
/// `years`: year for each band (length == band_count)
/// `params`: algorithm parameters
///
/// Returns flat Vec of pixel_count * 4, band-major:
///   [net_magnitude..., year..., rmse..., peak_to_trough_magnitude...]
///
/// Band 0 (net_magnitude) and bands 1/2 keep their original semantics for
/// back-compat. Band 3 (peak_to_trough_magnitude) is the standard LandTrendr
/// disturbance-depth statistic (fitted trough - fitted peak over the full
/// trajectory; 0.0 for monotonic rises; NaN for under-observed pixels). See
/// `pixel_summary` for the exact definition.
pub fn flat(
    data: &[f32],
    pixel_count: usize,
    band_count: usize,
    years: &[i32],
    params: &LandTrendrParams,
) -> Vec<f32> {
    // Fast path: zero-allocation workspace for supported time series lengths
    if band_count <= LT_MAX_N {
        let mut magnitude_out = vec![0.0f32; pixel_count];
        let mut year_out = vec![f32::NAN; pixel_count];
        let mut rmse_out = vec![0.0f32; pixel_count];
        let mut ptt_out = vec![f32::NAN; pixel_count];
        let mut ws = LandTrendrWorkspace::new();
        let mut ts = vec![0.0f32; band_count];

        for px in 0..pixel_count {
            for t in 0..band_count {
                ts[t] = data[t * pixel_count + px];
            }
            let (mag, yr, rmse, ptt) =
                pixel_summary(&ts, years, band_count, params, &mut ws);
            magnitude_out[px] = mag;
            year_out[px] = yr;
            rmse_out[px] = rmse;
            ptt_out[px] = ptt;
        }

        let mut out = Vec::with_capacity(pixel_count * 4);
        out.extend_from_slice(&magnitude_out);
        out.extend_from_slice(&year_out);
        out.extend_from_slice(&rmse_out);
        out.extend_from_slice(&ptt_out);
        return out;
    }

    // Fallback for time series longer than LT_MAX_N
    let mut magnitude_out = vec![0.0f32; pixel_count];
    let mut year_out = vec![f32::NAN; pixel_count];
    let mut rmse_out = vec![0.0f32; pixel_count];
    let mut ptt_out = vec![f32::NAN; pixel_count];
    let mut ts = vec![0.0f32; band_count];

    for px in 0..pixel_count {
        for t in 0..band_count {
            ts[t] = data[t * pixel_count + px];
        }
        let result = pixel(&ts, years, params);
        // Net spectral change: last fitted value minus first
        let fitted = &result.fitted;
        magnitude_out[px] = fitted[fitted.len() - 1] - fitted[0];
        // Peak-to-trough over the full fitted trajectory (standard disturbance
        // depth) — same definition as pixel_summary / extract.py.
        // Only computed for fitted pixels (>= min_observations_needed valid);
        // under-observed pixels leave ptt at the NaN init so they mask out.
        let n_valid_px = ts.iter().filter(|v| !v.is_nan()).count();
        if n_valid_px >= params.min_observations_needed {
            let mut peak_idx = 0usize;
            let mut trough_idx = 0usize;
            let mut peak_val = fitted[0];
            let mut trough_val = fitted[0];
            for i in 1..fitted.len() {
                let v = fitted[i];
                if v > peak_val { peak_val = v; peak_idx = i; }
                if v < trough_val { trough_val = v; trough_idx = i; }
            }
            ptt_out[px] = if trough_idx <= peak_idx { 0.0 } else { trough_val - peak_val };
        }
        // Year of greatest disturbance
        let mut max_mag: f32 = 0.0;
        let mut dist_year = f32::NAN;
        for seg in &result.segments {
            if seg.magnitude < max_mag {
                max_mag = seg.magnitude;
                dist_year = seg.start_year as f32;
            }
        }
        year_out[px] = dist_year;
        rmse_out[px] = result.rmse;
    }

    let mut out = Vec::with_capacity(pixel_count * 4);
    out.extend_from_slice(&magnitude_out);
    out.extend_from_slice(&year_out);
    out.extend_from_slice(&rmse_out);
    out.extend_from_slice(&ptt_out);
    out
}

/// LandTrendr per-year FTV (fitted-to-vertex) difference at a target year.
///
/// Returns `pixel_count` f32: `fitted[idx] - fitted[idx-1]`, where `idx` is the
/// position of `target_year` in `years`. This is eMapR's `getLtFtvDiff(year)`
/// signal (forestEnsembleFunctions L1795) — the fitted change *in* the target
/// year, which the forest-loss ensemble's `getLtProbabilities` stretches to a
/// 0-100 loss probability. It is distinct from `peak_to_trough` (band 3 of
/// `flat`), which is the largest disturbance over the *whole*
/// trajectory regardless of year. Signed in the input index's units; the caller
/// orients (loss sign) and stretches. NaN where `target_year` is absent, has no
/// prior year, or the pixel is under-observed.
pub fn ftvdiff_flat(
    data: &[f32],
    pixel_count: usize,
    band_count: usize,
    years: &[i32],
    target_year: i32,
    params: &LandTrendrParams,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; pixel_count];
    let idx = match years.iter().position(|&y| y == target_year) {
        Some(i) if i >= 1 => i,
        _ => return out, // target year absent or has no prior year -> all NaN
    };
    if band_count > LT_MAX_N {
        return out; // the validated fast-path fit only supports band_count <= LT_MAX_N
    }
    // Use the SAME fast-path fit as flat (despike -> vertices -> segment
    // fit, leaving the fitted trajectory in ws.fitted). pixel is a
    // separate, over-smoothing path — do not use it here.
    let mut ws = LandTrendrWorkspace::new();
    let mut ts = vec![0.0f32; band_count];
    for px in 0..pixel_count {
        for t in 0..band_count {
            ts[t] = data[t * pixel_count + px];
        }
        let n_valid = ts.iter().filter(|v| !v.is_nan()).count();
        if n_valid < params.min_observations_needed {
            continue; // leave NaN
        }
        pixel_core(&ts, years, band_count, params, &mut ws);
        out[px] = ws.fitted[idx] - ws.fitted[idx - 1];
    }
    out
}

/// Windowed LandTrendr loss magnitude around a target year (loss = a fitted DECREASE).
///
/// Sums the loss-direction per-year fitted drops `max(0, fitted[y-1] - fitted[y])` over
/// `[target_year - half_window, target_year + half_window]`. When a disturbance is fit as a
/// multi-year ramp, the single-year `ftvdiff_flat` reads only ~1/N of a loss
/// spread over N years (low recall); the window recovers the full magnitude. `half_window = 0`
/// is the single-year loss (clamped to >= 0). Returns a NON-NEGATIVE loss magnitude in the
/// input index's units (loss-down convention); the caller stretches to a loss probability.
/// Trade-off: a window gives up some year precision (a loss in target+-1 counts toward
/// target). NaN where target_year is absent / has no prior year / the pixel is under-observed.
pub fn loss_window(
    data: &[f32], pixel_count: usize, band_count: usize, years: &[i32],
    target_year: i32, half_window: usize, params: &LandTrendrParams,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; pixel_count];
    let idx = match years.iter().position(|&y| y == target_year) {
        Some(i) if i >= 1 => i,
        _ => return out,
    };
    if band_count > LT_MAX_N {
        return out;
    }
    let lo = idx.saturating_sub(half_window).max(1);
    let hi = (idx + half_window).min(band_count - 1);
    let mut ws = LandTrendrWorkspace::new();
    let mut ts = vec![0.0f32; band_count];
    for px in 0..pixel_count {
        for t in 0..band_count {
            ts[t] = data[t * pixel_count + px];
        }
        if ts.iter().filter(|v| !v.is_nan()).count() < params.min_observations_needed {
            continue;
        }
        pixel_core(&ts, years, band_count, params, &mut ws);
        let mut loss = 0.0f32;
        for y in lo..=hi {
            loss += (ws.fitted[y - 1] - ws.fitted[y]).max(0.0);
        }
        out[px] = loss;
    }
    out
}

#[cfg(feature = "python")]
mod python;
