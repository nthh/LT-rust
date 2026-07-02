//! Numerical regression tests on the bundled LT-GEE Fig 2.1 reference pixel
//! (center pixel of data/nbr_1984_2016.npz — mature conifer, clearcut 2001,
//! regrowth to 2016). The expected values are a snapshot of the kernel output
//! that was validated vertex-for-vertex against LandTrendr-IDL (GDL) and
//! native GEE LandTrendr — see README "Validation". Any change that perturbs
//! them is a behavior change and must be re-validated, not just committed.

use landtrendr::{flat, pixel, LandTrendrParams};

const YEARS: [i32; 33] = [
    1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995,
    1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
    2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
];

// Annual Landsat NBR, NaN = cloud gap (1997).
const NBR: [f32; 33] = [
    0.581876, 0.821040, 0.839013, 0.814932, 0.840668, 0.878496, 0.871875,
    0.813482, 0.840141, 0.794890, 0.813466, 0.811209, 0.860313, f32::NAN,
    0.836910, 0.829604, 0.825144, 0.541358, -0.051705, -0.007749, 0.113037,
    0.288136, 0.382658, 0.428383, 0.574156, 0.557002, 0.660480, 0.691947,
    0.699058, 0.727496, 0.768161, 0.783932, 0.758209,
];

#[test]
fn reference_pixel_snapshot() {
    let r = pixel(&NBR, &YEARS, &LandTrendrParams::default());

    let vertex_years: Vec<i32> = YEARS
        .iter()
        .zip(&r.is_vertex)
        .filter_map(|(&y, &v)| v.then_some(y))
        .collect();
    assert_eq!(
        vertex_years,
        vec![1984, 1985, 2000, 2001, 2002, 2008, 2016],
        "breakpoint years changed"
    );

    assert!(
        (r.rmse - 0.053808).abs() < 1e-5,
        "rmse changed: {}",
        r.rmse
    );

    const EXPECTED_FITTED: [f32; 33] = [
        0.581876, 0.710444, 0.722324, 0.734203, 0.746082, 0.757961, 0.769840,
        0.781719, 0.793598, 0.805477, 0.817356, 0.829235, 0.841114, 0.852993,
        0.864872, 0.876751, 0.888630, 0.541358, -0.051705, 0.050339, 0.152383,
        0.254428, 0.356472, 0.458516, 0.560560, 0.591347, 0.622133, 0.652920,
        0.683707, 0.714493, 0.745280, 0.776067, 0.806853,
    ];
    for (i, (got, want)) in r.fitted.iter().zip(&EXPECTED_FITTED).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "fitted[{i}] ({} in {}) changed: got {got}, expected {want}",
            YEARS[i],
            "reference pixel"
        );
    }
}

#[test]
fn flat_matches_pixel_on_reference() {
    // `flat` summary bands [net_mag, year, rmse, peak_to_trough] for the same pixel.
    let out = flat(&NBR, 1, YEARS.len(), &YEARS, &LandTrendrParams::default());
    let expected = [0.224977f32, 2001.0, 0.053808, -0.940335];
    assert_eq!(out.len(), 4);
    for (i, (got, want)) in out.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "flat band {i} changed: got {got}, expected {want}"
        );
    }
}

#[test]
fn flat_layout_is_band_major() {
    // Pin the data layout: `flat` reads data[t * pixel_count + px] (all pixels
    // for year 0, then year 1, ...) and returns 4 planar output bands of
    // pixel_count each. A 2-pixel stack where both pixels are the reference
    // series must reproduce the single-pixel result twice.
    let n = YEARS.len();
    let mut stack = vec![0.0f32; 2 * n];
    for t in 0..n {
        stack[t * 2] = NBR[t];
        stack[t * 2 + 1] = NBR[t];
    }
    let out = flat(&stack, 2, n, &YEARS, &LandTrendrParams::default());
    assert_eq!(out.len(), 8);
    let expected = [0.224977f32, 2001.0, 0.053808, -0.940335];
    for band in 0..4 {
        for px in 0..2 {
            let got = out[band * 2 + px];
            assert!(
                (got - expected[band]).abs() < 1e-4,
                "band {band} pixel {px}: got {got}, expected {}",
                expected[band]
            );
        }
    }
}

#[test]
fn under_observed_pixel_passes_through() {
    // Fewer valid observations than min_observations_needed → no fit: input
    // echoed back, no vertices, rmse 0.
    let years: [i32; 6] = [2000, 2001, 2002, 2003, 2004, 2005];
    let vals: [f32; 6] = [0.5, f32::NAN, f32::NAN, f32::NAN, f32::NAN, 0.4];
    let r = pixel(&vals, &years, &LandTrendrParams::default());
    assert!(!r.is_vertex.iter().any(|&v| v), "expected no vertices");
    assert_eq!(r.rmse, 0.0);
    assert_eq!(r.fitted[0], 0.5);
    assert!(r.fitted[1].is_nan());
}
