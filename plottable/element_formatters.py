import math

def _scientific_notation_ascii(significand, magnitude, fraction_digits=2, suppress_zero=False):
  if suppress_zero and magnitude == 0:
    return "{:.{precision}f}".format(significand, precision=fraction_digits)
  return "{:.{precision}f} x 10^{}".format(significand, magnitude, precision=fraction_digits)

def _scientific_notation_tex(significand, magnitude, fraction_digits=2, suppress_zero=False):
  if suppress_zero and magnitude == 0:
    return "{:.{precision}f}".format(significand, precision=fraction_digits)
  return "{:.{precision}f} \\times 10^{{{}}}".format(significand, magnitude, precision=fraction_digits)

def _scientific_notation_interval_tex(significand, interval, magnitude, fraction_digits=2):
  return "{:.{precision}f} \\pm {:.{precision}f} \\times 10^{{{}}}".format(significand, interval, magnitude, precision=fraction_digits)

def _kMB_notation(significand, magnitude, fraction_digits=2):
  units = ["", "k", "M", "B", "T"]
  try:
    unit = units[magnitude // 3]
  except:
    return _scientific_notation_ascii(siginificand, magnitude, fraction_digits)
  return "{:.{precision}f}{}".format(significand, unit, precision=fraction_digits)

def magnitude_significand(f, mod=1, negative_magnitude=True):
  magnitude = int(math.floor(math.log10(abs(f))))
  if not negative_magnitude:
    magnitude = max(0, magnitude)
  magnitude = mod * (magnitude // mod)
  significand = f * 10 ** (-magnitude)
  return magnitude, significand

def scientific_notation_interval_tex(f, interval, significant_digits=3):
  m, s = magnitude_significand(f)
  interval_s = interval * 10 ** (-m)
  return _scientific_notation_interval_tex(s, interval_s, m, significant_digits - 1)

def scientific_notation_tex(f, significant_digits=3):
  m, s = magnitude_significand(f)
  return _scientific_notation_tex(s, m, significant_digits - 1)

def scientific_notation_ascii(f, significant_digits=3):
  m, s = magnitude_significand(f)
  return _scientific_notation_ascii(s, m, significant_digits - 1)

def scientific_notation_tex(f, significant_digits=3):
  m, s = magnitude_significand(f)
  return _scientific_notation_tex(s, m, significant_digits - 1)

def engineering_notation_ascii(f, significant_digits=3):
  m, s = magnitude_significand(f, 3)
  significand_m, _ = magnitude_significand(s)
  return _scientific_notation_ascii(s, m, significant_digits - significand_m - 1, True)

def engineering_notation_tex(f, significant_digits=3):
  m, s = magnitude_significand(f, 3)
  significand_m, _ = magnitude_significand(s)
  return _scientific_notation_tex(s, m, significant_digits - significand_m - 1, True)

def kMB_notation(f, significant_digits=3):
  m, s = magnitude_significand(f, 3, False)
  significand_m, _ = magnitude_significand(s)
  return _kMB_notation(s, m, significant_digits - significand_m - 1)
