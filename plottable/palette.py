from scipy.interpolate import UnivariateSpline, SmoothBivariateSpline
import palettable
#from colormath.color_conversions import convert_color
#from colormath.color_objects import sRGBColor, LabColor
#from colormath.color_diff import delta_e_cie2000
import numpy as np
from math import pi
import skimage.color as color
from tqdm import tqdm
import json

#def _cart2polar_2pi(x, y):
#    """convert cartesian coordinates to polar (uses non-standard theta range!)
#    NON-STANDARD RANGE! Maps to ``(0, 2*pi)`` rather than usual ``(-pi, +pi)``
#    """
#    r, t = np.hypot(x, y), np.arctan2(y, x)
#    t += np.where(t < 0., 2 * np.pi, 0)
#    return r, t
#
#def ciede2000(c1, c2):
#  kL = 1
#  kC = 1
#  kH = 1
#  L1, a1, b1 = c1
#  L2, a2, b2 = c2
#  # (often denoted "prime" in the literature)
#  Cbar = 0.5 * (np.hypot(a1, b1) + np.hypot(a2, b2))
#  c7 = Cbar ** 7
#  G = 0.5 * (1 - np.sqrt(c7 / (c7 + 25 ** 7)))
#  scale = 1 + G
#  C1, h1 = _cart2polar_2pi(a1 * scale, b1)
#  C2, h2 = _cart2polar_2pi(a2 * scale, b2)
#  # recall that c, h are polar coordiantes.  c==r, h==theta
#
#  # cide2000 has four terms to delta_e:
#  # 1) Luminance term
#  # 2) Hue term
#  # 3) Chroma term
#  # 4) hue Rotation term
#
#  # lightness term
#  Lbar = 0.5 * (L1 + L2)
#  tmp = (Lbar - 50) ** 2
#  SL = 1 + 0.015 * tmp / np.sqrt(20 + tmp)
#  L_term = (L2 - L1) / (kL * SL)
#
#  # chroma term
#  Cbar = 0.5 * (C1 + C2)  # new coordiantes
#  SC = 1 + 0.045 * Cbar
#  C_term = (C2 - C1) / (kC * SC)
#
#  # hue term
#  h_diff = h2 - h1
#  h_sum = h1 + h2
#  CC = C1 * C2
#
#  dH = h_diff.copy()
#  dH[h_diff > np.pi] -= 2 * np.pi
#  dH[h_diff < -np.pi] += 2 * np.pi
#  dH[CC == 0.] = 0.  # if r == 0, dtheta == 0
#  dH_term = 2 * np.sqrt(CC) * np.sin(dH / 2)
#
#  Hbar = h_sum.copy()
#  mask = np.logical_and(CC != 0., np.abs(h_diff) > np.pi)
#  Hbar[mask * (h_sum < 2 * np.pi)] += 2 * np.pi
#  Hbar[mask * (h_sum >= 2 * np.pi)] -= 2 * np.pi
#  Hbar[CC == 0.] *= 2
#  Hbar *= 0.5
#
#  T = (1 -
#       0.17 * np.cos(Hbar - np.deg2rad(30)) +
#       0.24 * np.cos(2 * Hbar) +
#       0.32 * np.cos(3 * Hbar + np.deg2rad(6)) -
#       0.20 * np.cos(4 * Hbar - np.deg2rad(63))
#       )
#  SH = 1 + 0.015 * Cbar * T
#
#  H_term = dH_term / (kH * SH)
#
#  # hue rotation
#  c7 = Cbar ** 7
#  Rc = 2 * np.sqrt(c7 / (c7 + 25 ** 7))
#  dtheta = np.deg2rad(30) * np.exp(-((np.rad2deg(Hbar) - 275) / 25) ** 2)
#  R_term = -np.sin(2 * dtheta) * Rc * C_term * H_term
#
#  # put it all together
#  dE2 = L_term ** 2
#  dE2 += C_term ** 2
#  dE2 += H_term ** 2
#  dE2 += R_term
#  ans = np.sqrt(dE2)
#  return ans

def equidistant_points_2d_mesh(func, distance, n_samples, cyclic=(False, False), fit_boundary=(False, False), epsilon=1e-8, eta=1e-2, adaepsilon=1e-9, max_iteration=10000, clipping_value=100, update_dropout=0.2, burn_in_period=1000):
  """
    Compute an equidistant rectangular mesh on the given surface.

    Note: This function uses gradient descent optimization and thus
    it can be fallen into local minimum when the surface has winding shape.
     
    Parameters
    ----------
    func : function
        Continuous function which is the parametric represention
        of the surface.
        [0, 1] x [0, 1] -> A (subset of numpy array of shape (k,))

    distance : function
        Distance of two points in A.
        A x A -> [0, inf)

    (n, m) : (int, int)
        The shape of the resultant mesh array.

    (cyclic_x, cyclic_y): (bool, bool)
        Specify whether the curve has a circular topology.
        (i.e. func(0, *) = func(1, *))

    epsilon: float
        Small positive number used for numeric differentiation

    eta: float
        Small positive number used for parameter update.
        Larger value means faster but less stable convergence.

    adaepsilon: float
        Small positive number used for numerical stability of
        parameter update.

    max_iteration: int
        Maximum number of optimizing iteration.

    clipping_value : float
        The gradient clipping value ensuring convergence. Especially
        effective in cases where the speed of the parametric
        curve is uneven.

    Returns
    ----------
    s, t : np.array
      Coordinate matrices of shape (n, m) representing the computed mesh.
        forall i exists L
          forall j distance(f(s[i,j], t[i,j]), f(s[i+1,j], t_[i+1,j])) = L
        forall j exists L
          forall i distance(f(s[i,j], t[i,j]), f(s[i,j+1], t[i,j+1])) = L
        s[0, :] = t[:, 0] = 0
        s[n, :] = t[:, m] = 0 or 1

  """
  n, m = n_samples
  cyclic_x, cyclic_y = cyclic
  fit_boundary_x, fit_boundary_y = fit_boundary
  if cyclic_x:
    n = n + 1
  if cyclic_y:
    m = m + 1
  n_samples = n * m
  s, t = np.mgrid[0:1:n*1j, 0:1:m*1j]
  if cyclic_x:
    s[n-1, :] = 0
  if cyclic_y:
    t[:, m-1] = 0
  def pack(s, t):
    return np.hstack((s.flatten(), t.flatten()))
  def unpack(st):
    dims = len(st.shape)
    s = st[:n_samples]
    t = st[n_samples:]
    if dims == 2:
      s = s.reshape((n, m, -1))
      t = t.reshape((n, m, -1))
    elif dims == 1:
      s = s.reshape((n, m))
      t = t.reshape((n, m))
    return s, t

  def loss(s, t):
    loss = 0
    n = s.shape[0]
    m = s.shape[1]
    points = func(s, t)
    # x
    interval_distances = distance(points[:-1], points[1:])
    for i in range(1,n//2):
      u = interval_distances[i:, :] - interval_distances[:-i, :]
      loss += np.sum(u**2, axis=(0, 1))
    # y
    interval_distances = distance(points[:,:-1], points[:,1:])
    for i in range(1,m//2):
      u = interval_distances[:, i:] - interval_distances[:, :-i]
      loss += np.sum(u**2, axis=(0, 1))
    return loss, interval_distances


  # AdaGrad initialization
  h = np.zeros((2 * n_samples,)) + adaepsilon

  dI = epsilon * np.eye(2 * n_samples)

  for i in range(max_iteration):
    L, distances = loss(s, t)
    # numerical differentiation
    # (d/dt L)(t)
    stm = pack(s, t).reshape((-1, 1)) + dI
    sm, tm = unpack(stm)
    Lm, _ = loss(sm, tm)
    dLdt = (Lm - L) / epsilon
    max_distance = np.max(distances)
    dLdt_normalized = dLdt / max_distance

    gs, gt = unpack(dLdt_normalized)
    if fit_boundary_x:
      gs[0, :] = 0
      gs[n-1, :] = 0
    else:
      gs[0, 0] = 0
      gs[n-1, 0] = 0
    if fit_boundary_y:
      gt[:, 0] = 0
      gt[:, m-1] = 0
    else:
      gt[0, 0] = 0
      gt[0, m-1] = 0
    if cyclic_x:
      gs[0, :] = gs[0, :] + gs[n-1, :]
      gs[n-1, :] = gs[0, :]
      gt[0, :] = gt[0, :] + gt[n-1, :]
      gt[n-1, :] = gt[0, :]
    if cyclic_y:
      gs[:, 0] = gs[:, 0] + gs[:, m-1]
      gs[:, m-1] = gs[:, 0]
      gt[:, 0] = gt[:, 0] + gt[:, m-1]
      gt[:, m-1] = gt[:, 0]
    g = pack(gs, gt)

    # stochastic update
    mask = np.random.choice(a=[0, 1], size=g.shape, p=[update_dropout, 1-update_dropout])
    g *= mask

    # burn-in period
    g *= np.minimum(1, i/burn_in_period)

    # AdaGrad update rule
    h = h + g ** 2
    st = pack(s, t) - eta * g / np.sqrt(h)
    s, t = unpack(st)
  print(L)
  print(s)
  print(t)
  return s, t

def equidistant_points_1d(func, distance, n_samples, cyclic=False, epsilon=1e-9, eta=1e-2, adaepsilon=1e-9, max_iteration=3000, clipping_value=10, update_dropout=0.2):
  """
    Sample equidistant points on the given curve.

    Note: This function uses gradient descent optimization and thus
    it can be fallen into local minimum when the curve has winding shape.
     
    Parameters
    ----------
    func : function
        continuous function which is the parametric represention of the curve
        [0, 1] -> A (A: Riemannian manifold e.g. numpy array of shape (k,))

    distance : function
        distance of two points in A
        A x A -> R

    n_samples : int
        the curve will split into n_samples intervals

    cyclic: boolean
        whether the curve has a circular topology
        (i.e. func(0) = func(1))

    epsilon: float
        small positive number used for numeric differentiation

    eta: float
        small positive number used for parameter update

    max_iteration: int
        maximum number of optimizing iteration

    clipping_value : float
        the gradient clipping value ensuring convergence especially
        in cases where the speed of the parametric curve is uneven.

    Returns
    ----------
    t : np.array
             [t_0, t_1, ... , t_{n_samples}]
              s.t. exists L forall i distance(f(t_i), f(t_{i+1})) = L
                   t_0 = 0
                   t_{n_samples} = 0 (cyclic) 1 (otherwise)

  """
  if cyclic:
    n_samples = n_samples + 1
  dI = epsilon * np.eye(n_samples)
  t = np.linspace(0, 1, n_samples)
  if cyclic:
    t[-1] = 0

  def loss(t):
    # t is a vector [t_0, ..., t_{n_samples}]
    interval_distances = distance(func(t[:-1]), func(t[1:]))
    loss = 0
    n = t.shape[0]
    # iterate over 2-combinations
    for i in range(1,n//2):
      u = interval_distances[i:] - interval_distances[:-i]
      loss += np.sum(u**2, axis=0)
    return loss, interval_distances

  # AdaGrad initialization
  h = np.zeros((n_samples - 2,)) + adaepsilon

  for i in range(max_iteration):
    L, distances = loss(t)
    # numeric differentiation
    # (d/dt L)(t)
    Lm, _ = loss(t.reshape((-1, 1)) + dI)
    dLdt = (Lm - L) / epsilon
    max_distance = np.max(distances)
    dLdt_normalized = dLdt / max_distance
    # gradient descent with clipping
    norm = np.linalg.norm(dLdt_normalized)
    if norm > clipping_value:
      dLdt_normalized = clipping_value * dLdt / norm
    g = dLdt_normalized[1:-1]
    # stochastic update
    mask = np.random.choice(a=[0, 1], size=g.shape, p=[update_dropout, 1-update_dropout])
    g *= mask
    # AdaGrad update rule
    h = h + g ** 2
    t[1:-1] = t[1:-1] - eta * g / np.sqrt(h)
  return t

def test_equidistant_points_1d(n=4):
  def parabola(t):
    return np.stack((t, t**2))
  def euclidean(p, q):
    return np.linalg.norm(p - q, axis=0)
  t = equidistant_points_1d(parabola, euclidean, n)
  return t, euclidean(parabola(t[1:]), parabola(t[:-1]))

def test_equidistant_points_2d(n=4, m=4, iter=3000):
  def parabola(s, t):
    return np.stack((s, t, s**2 * t**2))
  def euclidean(p, q):
    return np.linalg.norm(p - q, axis=0)
  t = equidistant_points_2d_mesh(parabola, euclidean, (n, m), max_iteration=iter)
  return t

def test_ciede2000():
  c1 = np.array([30, 14, 18])
  c2 = np.array([50, 20, 48])
  return ciede2000(c1, c2)

def test_color_parabola(n=4, iter=3000):
  def curve(t):
    r = 10 + 30 * (1 - t) ** 2
    L = 65 + 0*t#+ 30 * t
    theta = - pi * t
    return np.stack((L, r*np.cos(theta), r*np.sin(theta)))
  t = equidistant_points_1d(curve, ciede2000, n, max_iteration=iter)
  for tk in t:
    lab = LabColor(*curve(tk))
    srgb = convert_color(lab, sRGBColor)
    print("<div style='height:120px;width:120px;background:{}'></div>".format(srgb.get_rgb_hex()))
  return t, ciede2000_2(curve(t[1:]), curve(t[:-1]))

def test_color_2d(n=4, m=4, iter=3000):
  def surface(s, t):
    z = s**2 * t**2
    return np.stack((30 + 30 * z, 50 * s, 50 * t))
  s, t = equidistant_points_2d_mesh(surface, ciede2000, (n, m), max_iteration=iter)
  print("<table>")
  for si, ti in zip(s, t):
    print("<tr>")
    for sij, tij in zip(si, ti):
      lab = LabColor(*surface(sij, tij))
      srgb = convert_color(lab, sRGBColor)
      print("<td style='height:120px;width:120px;background:{}'></td>".format(srgb.get_rgb_hex()))
    print("</tr>")
  print("</table>")

def parse_hex(hex_string):
  if hex_string[0] == "#":
    hex_string = hex_string[1:]
  if len(hex_string) == 3:
    rgb = [int(c, 16) for c in hex_string]
  elif len(hex_string) == 6:
    rgb = [int(hex_string[i:i+2], 16) for i in range(0, 6, 2)]
  else:
    raise ValueError(hex_string)
  return np.asarray(rgb) / 255.

def stringify_hex(color):
  color = np.asarray(0xff * np.minimum(color, 1), dtype=np.uint8)
  return "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

def get_color_map_1d(anchors):
  n = len(anchors)
  anchors = anchors.reshape((1, -1, 3))
  x = np.mgrid[0:1:n*1j]
  lab = color.rgb2lab(anchors)
  lch = color.lab2lch(lab)
  lch = lch.reshape((-1, 3))
  L = UnivariateSpline(x, lch[:,0])
  C = UnivariateSpline(x, lch[:,1])
  H = UnivariateSpline(x, lch[:,2])
  def curve(t):
    LCH = np.stack((L(t), C(t), H(t)), axis=-1)
    return color.lch2lab(LCH)
  return curve

def get_color_map_2d(anchors):
  n, m, ch = anchors.shape
  assert ch == 3
  s, t = np.mgrid[0:1:n*1j, 0:1:m*1j]
  lab = color.rgb2lab(anchors)
  lch = color.lab2lch(lab)
  s = s.ravel()
  t = t.ravel()
  h = lch[:,:,2]
  for j in range(1, m):
    for i in range(n):
      while True:
        diff = h[i, j] - h[i, j-1]
        if diff > pi:
          h[i, j] -= 2*pi
        elif diff < -pi:
          h[i, j] += 2*pi
        else:
          break
  h_avg = np.average(h, weights=lch[:,:,1], axis=1)
  reverse = np.zeros(h_avg.shape, dtype=np.uint8)
  reverse[1:] = h_avg[1:] < h_avg[:-1]
  reverse=np.cumsum(reverse)
  print(reverse)
  h += reverse.reshape((-1, 1)) * 2*pi
  print(h)
  l = lch[:,:,0].ravel()
  c = lch[:,:,1].ravel()
  h = h.ravel()
  kx = min(1, n-1)
  L = SmoothBivariateSpline(s, t, l, kx=kx)
  C = SmoothBivariateSpline(s, t, c, kx=kx)
  H = SmoothBivariateSpline(s, t, h, kx=kx)
  def surface(s, t):
    LCH = np.stack((L(s, t, grid=False), C(s, t, grid=False), H(s, t, grid=False)), axis=-1)
    return color.lch2lab(LCH)
  return surface

def gen_palettable_colors():
  colors = {}
  modules = [
      palettable.cartocolors.diverging,
      palettable.cartocolors.qualitative,
      palettable.cartocolors.sequential,
      palettable.cmocean.diverging,
      palettable.cmocean.sequential,
      palettable.colorbrewer.diverging,
      palettable.colorbrewer.qualitative,
      palettable.colorbrewer.sequential,
      palettable.matplotlib,
      palettable.mycarta,
      palettable.tableau,
      palettable.wesanderson ]
  modules = modules[3:8]
  for m in modules:
    print("generating", m.__name__)
    for k, v in m.__dict__.items():
      if k[-2:] == "_5":
        name = m.__name__ + "." + k[:-2]
        print("generating", name)
        palette_colors = v.colors
        palette_colors = np.asarray(palette_colors)/255
        color_curve = get_color_map_1d(palette_colors)
        obj = {}
        colors[name] = obj
        obj["curve"] = color_curve
        for i in tqdm(range(4, 7)):
          t = equidistant_points_1d(color_curve, color.deltaE_ciede2000, i)
          rgb = color.lab2rgb(color_curve(t).reshape((1, -1, 3))).reshape((-1, 3))
          obj[i] = rgb
  return colors

def gen_2d_colors():
  colors = {}
  palettes_list = {
    "thermal": [
      palettable.colorbrewer.sequential.RdPu_5,
      palettable.cmocean.sequential.Thermal_5_r],
    "ATS": [
      palettable.cmocean.sequential.Amp_5,
      palettable.cmocean.sequential.Turbid_5,
      palettable.cmocean.sequential.Speed_5],
    "Yl-BrGnBu": [
      palettable.colorbrewer.sequential.YlOrBr_5,
      palettable.colorbrewer.sequential.YlGn_5,
      palettable.colorbrewer.sequential.YlGnBu_5],
    "GnBuBr": [
      palettable.colorbrewer.sequential.YlGn_5,
      palettable.colorbrewer.sequential.YlGnBu_5,
      palettable.colorbrewer.sequential.YlOrBr_5],
    "GBBG": [
      palettable.colorbrewer.sequential.YlGn_5,
      palettable.colorbrewer.sequential.YlGnBu_5,
      palettable.colorbrewer.sequential.YlOrBr_5],
    "PuBuGn-YlOrRd": [
      palettable.colorbrewer.sequential.PuBuGn_5,
      palettable.colorbrewer.sequential.YlOrRd_5]}
  flags = {
    "thermal": ((False, False), (False, False)),
    "ATS":  ((False, False), (False, False)),
    "Yl-BrGnBu": ((False, False), (True, True)),
    "GnBuBr":  ((False, False), (False, False)),
    "GBBG":  ((True, False), (False, True)),
    "PuBuGn-YlOrRd":  ((False, False), (False, True))
   }

  for name, palettes in palettes_list.items():
    palette_colors = [p.colors for p in palettes]
    palette_colors = np.asarray(palette_colors)/255
    color_surface = get_color_map_2d(palette_colors)
    obj = {}
    colors[name] = obj
    flag = flags[name]
    for i in tqdm(range(4, 7)):
      s, t = equidistant_points_2d_mesh(color_surface, color.deltaE_ciede2000, (i, i), cyclic=flag[0], fit_boundary=flag[1])
      rgb = color.lab2rgb(color_surface(s, t))
      obj[i] = rgb
  return colors

def ffcolor(ar):
  return np.round(0xff * np.minimum(1, ar)).astype(np.uint8)

def to_html_table(schemes, n=4):
  for name, lis in schemes.items():
    print("<h2>", name, "</h2>")
    print("<table>")
    for r in lis[n]:
      print("<tr>")
      for c in r:
        print("<td style='height:120px;width:120px;background:{}'></td>".format(stringify_hex(c)))
      print("</tr>")
    print("</table>")

def to_colorschemes_js(schemes):
  schemes_jsobj = {name: 
      {number: ["rgb({},{},{})".format(a[0], a[1], a[2]) for a in ffcolor(color)]
        for number, color in colors.items()} for name, colors in schemes.items()}
  return json.dumps(schemes_jsobj)

def test_color_1d(anchor_hexcolor, n=4, iter=3000):
  anchors = np.asarray([parse_hex(c) for c in anchor_hexcolor])
  curve = get_color_map_1d(anchors)
  t = equidistant_points_1d(curve, color.deltaE_ciede2000, n, max_iteration=iter)
  rgb = color.lab2rgb(curve(t).reshape((1, -1, 3))).reshape((-1, 3))
  for c in rgb:
    print("<div style='height:120px;width:120px;background:{}'></div>".format(stringify_hex(c)))
  return rgb

def test_color_2d(anchor_hexcolor, n=4, iter=3000):
  anchors = np.asarray([parse_hex(c) for c in anchor_hexcolor])
  curve = get_color_map_1d(anchors)
  t = equidistant_points_1d(curve, color.deltaE_ciede2000, n, max_iteration=iter)
  rgb = color.lab2rgb(curve(t).reshape((1, -1, 3))).reshape((-1, 3))
  for c in rgb:
    print("<div style='height:120px;width:120px;background:{}'></div>".format(stringify_hex(c)))
  return rgb
