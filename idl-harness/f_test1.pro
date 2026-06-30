;+
; F_TEST1 shim: CDF of the F-distribution, P(F <= f) for (df1, df2) dof.
;
; LandTrendr's calc_fitting_stats3 computes p_of_f = 1 - f_test1(f, df1, df2),
; i.e. the upper-tail F p-value. f_test1 is not in the public LandTrendr-2012
; source (it came from the original OSU IDL environment). It is the standard
; analytic F cumulative distribution, expressed via the regularized incomplete
; beta function:
;     F_CDF(f; d1, d2) = I_{ d1 f / (d1 f + d2) }( d1/2, d2/2 )
; so this reproduces the intended statistic exactly rather than approximating.
; Verified: ibeta(2.5, 5.0, 0.5) = 0.83581 = F_CDF(2.0; 5,10), p = 0.164.
;-
function f_test1, f, df1, df2
  compile_opt idl2
  ff = double(f)
  if ff le 0d then return, 0d
  d1 = double(df1)
  d2 = double(df2)
  z = d1 * ff / (d1 * ff + d2)
  return, ibeta(d1 / 2d, d2 / 2d, z)
end
