;+
; REGRESS shim for the GDL headless build (which omits regress.pro).
;
; LandTrendr's tbcd_v2 calls REGRESS in exactly one shape, in three places:
;     r = regress(x_1d, y_1d, yfit=yfit)
; i.e. a single independent variable, unit weights, returning the slope and
; (via the YFIT keyword) the fitted line. For that case this is mathematically
; identical to IDL/GDL's library REGRESS, so it preserves the algorithm rather
; than approximating it. Extra keywords are accepted/ignored for signature
; compatibility.
;-
function regress, x, y, weights, const=const, yfit=yfit, $
                  mcorrelation=mcorrelation, sigma=sigma, ftest=ftest, $
                  chisq=chisq, status=status, _ref_extra=ex
  compile_opt idl2

  xx = double(reform(x))
  yy = double(reform(y))
  n  = n_elements(yy)

  sx  = total(xx)      & sy  = total(yy)
  sxx = total(xx * xx) & sxy = total(xx * yy)
  denom = n * sxx - sx * sx
  slope = (denom ne 0d) ? (n * sxy - sx * sy) / denom : 0d
  const = (sy - slope * sx) / n

  yfit  = float(slope * xx + const)
  status = 0L
  return, float(slope)
end
