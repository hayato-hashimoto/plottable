import math
import numpy as np
import pandas as pd
import xarray as xr
import pickle

from . import palette

pi = math.pi
columns = object()
types = ["bar", "bubble", "pie", "square", "line"]

with open("colors2d.pkl", "rb") as f:
  colors2d = pickle.load(f)
with open("colors.pkl", "rb") as f:
  colors = pickle.load(f)

def get_colormap(name, x):
  if isinstance(x, pd.MultiIndex):
    return {tuple([x.levels[i][c] for i, c in enumerate(code)]):
        palette.stringify_hex(colors2d[name][4][code]) for code in zip(*x.codes)}
  else:
    name = "palettable.colorbrewer.sequential.BuGn"
    return {k: palette.stringify_hex(colors[name][4][i]) for i, k in enumerate(x)}
    

def extract_dict(dic, criterion):
  extract = {}
  rest = {}
  for k, v in dic.items():
    if criterion(v):
      extract[k]=v
    else:
      rest[k]=v
  return extract, rest

def extract(seq, criterion):
  extract = []
  rest = []
  indices = []
  for i, a in enumerate(seq):
    if criterion(a):
      indices.append(i)
      extract.append(a)
    else:
      rest.append(a)
  return extract, rest, indices

def inject_dict(a, b):
  return {**a, **b}

def inject(a, b, indices):
  a = list(a)
  b = list(b)
  n = len(a) + len(b)
  ret = []
  for i in range(n):
    if i in indices:
      ret.append(b.pop(0))
    else:
      ret.append(a.pop(0))
  return ret

def pack(seq, mappings):
  keys = list(mappings.keys())
  return ((len(seq), keys), seq + [mappings[k] for k in keys])

def unpack(tags, packings): 
  l, keys = tags
  seq = packings[:l]
  mappings = {k: packings[l+i] for i, k in enumerate(keys)}
  return seq, mappings

def xr_join(sep, args):
  strings, xarrays, positions = extract(args, lambda x: isinstance(x, str))
  def join(*args):
    return sep.join(inject(args, strings, positions))
  return xr.apply_ufunc(join, *xarrays, vectorize=True)

def xr_concat(*args):
  return xr_join("", args)

def xr_format(format_string, *args, **kwargs):
  string_args, xarray_args, string_indices = extract(args, lambda x: isinstance(x, str))
  string_kwargs, xarray_kwargs = extract_dict(kwargs, lambda x: isinstance(x, str))
  xr_tags, xr_packs = pack(xarray_args, xarray_kwargs)
  def format(*packs):
    xr_args, xr_kwargs = unpack(xr_tags, packs)
    args = inject(xr_args, string_args, string_indices)
    kwargs = inject_dict(xr_kwargs, string_kwargs)
    return format_string.format(*args, **kwargs)
  return xr.apply_ufunc(format, *xr_packs, vectorize=True)

def xr_to_string(x):
  def convert(ar):
    return np.asarray(ar, dtype=np.unicode)
  assert isinstance(x, (xr.DataArray, xr.Dataset))
  return xr.apply_ufunc(convert, x)

def xr_sum_string(x, dim=None):
  def sum_string(ar, axis=None):
    # np.defchararray.add.reduce is not defined
    return np.add.reduce(np.asarray(ar, dtype=np.object), axis=axis)
  assert isinstance(x, (xr.DataArray, xr.Dataset))
  return x.reduce(sum_string, dim=dim)

def miniplot(plot_type, value, axis=None, **options):
  require_1d = plot_type == "line"
  if isinstance(value, dict):
    output_type="dict"
    scalar = False
    if any(np.asarray(v).shape==() for v in value.values()):
      value = {k: [v] for k, v in value.items()}
      scalar = True
    value = pd.DataFrame(value).to_xarray()
    if scalar:
      value = value.isel({"index": 0}, drop=True)
    if require_1d and axis is None:
      axis="index"
      if scalar:
        raise ValueError("{} plot requires 1-d array element".format(plot_type))
  elif isinstance(value, pd.Series):
    if require_1d and axis is None:
      axis = value.index.name
    value = value.to_xarray()
    output_type = "pd.Series"
  elif isinstance(value, pd.DataFrame):
    if require_1d and axis is None:
      axis = value.index.name
    value = value.to_xarray()
    output_type = "pd.DataFrame"
  elif isinstance(value, (xr.DataArray, xr.Dataset)):
    if require_1d and axis is None:
      axis = next(value.coords.__iter__())
    output_type = "xarray"
  else:
    value = np.asarray(value)
    if require_1d and axis is None:
      axis = "dim_{}".format(len(value.shape) - 1)
    value = xr.DataArray(value)
    output_type = "numpy"

  if axis == columns:
    assert isinstance(value, xr.Dataset) # or converted from pd.DataFrame, dict
    labels = [v for v in value.data_vars]
    concat_index = pd.Index(labels)
    value = xr.concat([value[v] for v in value.data_vars], concat_index).rename(concat_dim="*proportion*")
    axis="*proportion*"

  if plot_type == 'bar':
    ret = miniplot_bar(value, **options)
  elif plot_type == 'bubble':
    ret = miniplot_bubble(value, **options)
  elif plot_type == 'pie':
    ret = miniplot_pie(value, **options)
  elif plot_type == 'square':
    ret = miniplot_square(value, axis=axis, **options)
  elif plot_type == 'line':
    ret = miniplot_line(value, axis=axis, **options)

  if output_type == "dict":
    if isinstance(ret, xr.DataArray):
      if ret.shape == ():
        return ret.item()
      return ret.to_dataframe(name="*proportion*").to_dict()
    elif not "index" in ret.coords or ret.coords["index"].ndim == 0:
      return {v: ret[v].data.item() for v in ret.data_vars}
    else:
      return ret.to_dataframe().to_dict()
  elif output_type == "pd.Series":
    return ret.to_series()
  elif output_type == "pd.DataFrame":
    if isinstance(ret, xr.DataArray):
      return ret.to_series()
    else:
      return ret
  elif output_type == "numpy":
    return ret.data
  else:
    return ret

def miniplot_bar(value, output="html", norm=None, color="#000000", width=100, height=20):
  if norm is None:
    norm = value.max()
  v = value/norm
  if output == "html":
    rs = xr_format('<div style="display:inline-block;min-width:{w}"><div style="display:inline-block;height:{h};line-height:{h};vertical-align:middle;width:{barw};background-color:{c}"></div></div>', h=height, w=width, barw=width*v, c=color)
  return rs

def miniplot_bubble(value, output="svg", color="#000000", reference=True, reference_color=None, shape="rect", scale="area", size=30, norm=1, **kwargs):
  v = 100 * math.sqrt(value / norm)
  if reference_color is None:
    reference_color = color
  if output == "svg":
    rs = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px" y="0px" width="{size}px" height="{size}px" viewBox="-110 -110 220 220">'.format(size=size)
    if shape == "rect":
      rs += '<rect fill="{c}" x="{a}" y="{a}" width="{b}" height="{b}"/>'.format(a=-v,b=2*v,c=color)
      if reference:
        rs += '<rect fill="none" style="stroke:{c};stroke-dasharray: 10 10;stroke-width: 3;" x="-110" y="-110" width="220" height="220"/>'.format(c=reference_color)
    elif shape == "circle":
      rs += '<circle cx="0" cy="0" r="{a}" fill="{c}"/>'.format(a=v, c=color)
      if reference:
        rs += '<circle cx="0" cy="0" r="100" fill="none" style="stroke:{c};stroke-dasharray: 10 10;stroke-width: 10;"/>'
    rs += "</svg>"
    return rs

def miniplot_pie(value, output="svg", color="#ee3300", fill_color="#ffaa88", reference=True, reference_color="#cccccc", size=30, **kwargs):
  """
  pie chart 
  """
  if output == "svg":
    rs = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px" y="0px" width="{size}px" height="{size}px" viewBox="-120 -120 240 240">'.format(size=size)
    rs += '<circle cx="0" cy="0" r="100" fill="none" stroke="{c}" stroke-width="30"/>'.format(c=reference_color)
    if fill_color is not None:
      rs += '<path d="M 0 0 L 0 -100 A 100 100 0 0 {flag} {x} {y} Z" fill="{c}" stroke="none"/>'.format(
        flag=1 if value<0.5 else 0, # counterclockwise
        x=100*math.sin(value*2*pi),
        y=-100*math.cos(value*2*pi),
        c=fill_color)
    rs += '<path d="M 0 -100 A 100 100 0 0 {flag} {x} {y}" fill="none" stroke="{c}" stroke-width="30"/>'.format(
      flag=1 if value<0.5 else 0, # counterclockwise
      x=100*math.sin(value*2*pi),
      y=-100*math.cos(value*2*pi),
      c=color)
    rs += "</svg>"
    return rs

def miniplot_square(value, output="svg", color="#ff4400", color_others="#eeeeee", colormap=None, width=None, height=30, nx=10, ny=4, misc_cutoff=1, allocation_method="Webster", randomize=False, spacings=0.1, axis=None, **kwargs):
  """
  Square chart 
  """
  if width is None:
    width = height * nx / ny
  if height is None:
    height = width * ny / nx
  if width / nx < height / ny:
    ds = 200 / nx
    w = 100
    h = 100 * height / width
  else:
    ds = 200 / ny
    w = 100 * width / height
    h = 100
  n = nx * ny
  if axis in value.indexes and isinstance(value.indexes[axis], pd.MultiIndex):
    misc_label = ["*"] * value.indexes[axis].nlevels
    misc_label[0] = "misc"
    misc_label = tuple(misc_label)
  else:
    misc_label = "misc"
  if axis is None:
    value = xr.concat([value, 1-value], pd.Index(["value", "others"], name="*proportion*"))
    axis = "*proportion*"
    if colormap is None:
      colormap = {"value": color, "others": color_others}
  else:
    if colormap is None:
      colormap = "GnBuBr"
    cutoff = misc_cutoff * value.sum(dim=axis) / n
    miscs = value < cutoff
    if len(miscs) > 0:
      pass
      #misc_ds = (value * miscs).sum(dim=axis).isel(drop=True).expand_dims(axis)
      #misc_ds[axis] = [misc_label]
      #value = xr.concat([value, misc_ds], dim=axis)
      #value = (1 - miscs) * value
      #print(value)
  x = value[axis]
  y = value
  if isinstance(colormap, str):
    colormap = get_colormap(colormap, value.indexes[axis])

  if allocation_method == "Adams":
    divisor = lambda n: n
  elif allocation_method == "Webster":
    divisor = lambda n: n + 0.5
  elif allocation_method == "D'Hondt":
    divisor = lambda n: n + 1
  else:
    raise ValueError("Unknown allocation method {}.".format(allocation_method))

  n_seats = xr.zeros_like(value, dtype=int)
  value_tmp = value / divisor(n_seats)
  is_dataarray = isinstance(value, xr.DataArray)
  for i in range(n):
    winner = value_tmp.argmax(dim=axis)
    if is_dataarray:
      win = {axis: winner}
      n_seats[win] += 1
      value_tmp[win] = value[win] / divisor(n_seats[win])
    else:
      for name in n_seats.data_vars:
        win = {axis: winner[name]}
        ns = n_seats[name]
        vt = value_tmp[name]
        v = value[name]
        a = ns[win]
        a += 1
        ns[win] = a
        vt[win] = v[win] / divisor(ns[win])

  def isel_indexer(ar, axis, indexer):
    ret = [slice(None)] * ar.ndim
    ret[axis] = indexer
    return tuple(ret)

  def broadcast_at(ar, axis, ndim):
    size = [1] * ndim
    size[axis] = -1
    return ar.reshape(size)

  def allocate_seats(ar, axis=-1):
    m = ar.shape[axis] # number of categories
    shape = list(ar.shape)
    shape[axis] = n
    seats = np.zeros(shape, dtype=int)
    indices = np.cumsum(ar, axis=axis)
    seat_number = broadcast_at(np.arange(n), axis=axis, ndim=seats.ndim)
    for i in range(m):
      seats += seat_number < indices[isel_indexer(indices, axis, slice(i, i+1))]
    seats = m - seats
    if randomize:
      seats = np.swapaxes(seats, 0, axis)
      np.random.shuffle(seats)
      seats = np.swapaxes(seats, 0, axis)
    return seats

  value_x = value[axis].data
  def to_svg_rectangles(seat, x, y):
    return '<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{c}" rx="4" ry="4"/>'.format(x=(x - nx/2 + spacings/2)*ds, y=(y - ny/2 + spacings/2)*ds, size=(ds - ds * spacings), c=colormap[value_x[seat]])

  def reduce_pipeline(ar, axis=-1):
    # seat allocation & convert to svg string
    seats = allocate_seats(ar, axis=axis)
    x, y = np.mgrid[:nx,:ny]
    x = broadcast_at(x, axis=axis, ndim=seats.ndim)
    y = broadcast_at(y, axis=axis, ndim=seats.ndim)
    rectangles = np.vectorize(to_svg_rectangles)(seats, x, y)
    ret = np.add.reduce(np.asarray(rectangles, dtype=object), axis=axis)
    return ret
  
  if output == "svg":
      start_tag = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px" y="0px" width="{width}px" height="{height}px" viewBox="{a} {b} {c} {d}">'.format(width=width, height=height, a=-w, b=-h, c=2*w, d=2*h)
      rectangles = n_seats.reduce(reduce_pipeline, dim=axis)
      return xr_concat(start_tag, rectangles, "</svg>")

def miniplot_line(value, output="svg", color="#ee3300", fill_color="#ffaa88", width=80, height=30, xlim=None, ylim=None, draw_area=True, axis=None, **kwargs):
  h = 100
  w = width / height

  x = value[axis]
  y = value
  if xlim is None:
    xlim = (x.min(), x.max())
  if ylim is None:
    ylim = (1.1*value.min().clip(max=0), 1.1*value.max().clip(min=0))

  if output == "svg":
    h = 100
    w = 100 * width / height
    start_tag = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px" y="0px" width="{width}px" height="{height}px" viewBox="{a} {b} {c} {d}">'.format(width=width, height=height, a=-5, b=-5, c=w+5, d=h+5)
    miniplot_x = xr_to_string(w * (x - xlim[0]) / (xlim[1] - xlim[0]))
    miniplot_y = xr_to_string(h - h * (y - ylim[0]) / (ylim[1] - ylim[0]))
    pathstring = xr_join(" ", [miniplot_x, miniplot_y])
    # join strings along axis
    start = xr_join(" ", ["M", pathstring.isel({x.name: 0})])
    pathstring = xr_sum_string(xr_concat(" L ", pathstring), dim=axis)
    pathstring_line = xr_concat(start, pathstring)
    zero_y = (h - h*-ylim[0] / (ylim[1]-ylim[0]))
    start = xr_format("M 0 {z}", z=zero_y)
    end = xr_format("L {w} {z} Z", w=w, z=zero_y)
    pathstring_area = xr_join(" ", [start, pathstring, end])
    line_plot = xr_format('<path d="{value}" fill="none" stroke-width="10" stroke="{c}" stroke-linecap="round"/>', c=color, value=pathstring_line)
    if draw_area:
      template = '<path d="{{value}}" fill="{c}"/>'.format(c=fill_color)
      area_plot = xr_format(template, value=pathstring_area)
      return xr_concat(start_tag, area_plot, line_plot, "</svg>")
    else:
      return xr_concat(start_tag, line_plot, "</svg>")
  else:
    raise NotImplementedError("output == 'svg'")
