import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from . import palette
from .html import HTMLFormatter

import pickle

with open("colors2d.pkl", "rb") as f:
  colors2d = pickle.load(f)

def get_colormap(name, labels):
  tags = get_colormap
  def cmap(*labels):
    ijk = [tag.index(label) for tag, label in zip(tags, labels)]
    return colors2d[name][5][ijk]
  return cmap

def expand_ellipsis(a, l):
  if Ellipsis not in a:
    return a[:l]
  eidx = a.index(Ellipsis)
  suffixes = len(a) - eidx
  def at(i):
    if i < eidx:
      return a[i]
    elif l - i < suffixes:
      return a[-l+i]
    else:
      return a[eidx-1]
  return [at(i) for i in range(l)]

def plot(data, columns, rows, color, scale, label, detailk
def plot_table(data, shapes="s", scale="area", colors="Yl-BrGnBu", cellspacing=0.3, ax=None, row_styles=[None, ...], column_styles=["marker-color", ...], data_styles=None):
  is_multi_c = isinstance(data.columns, pd.MultiIndex)
  is_multi_r = isinstance(data.index, pd.MultiIndex)
  def cmap(a, b):
    return palette.stringify_hex(colors2d[colors][4][a][b])
  if is_multi_c:
    cnlevels = data.columns.nlevels
  else:
    cnlevels = 1
  if is_multi_r:
    rnlevels = data.index.nlevels
  else:
    rnlevels = 1
  column_styles = expand_ellipsis(column_styles, cnlevels)
  row_styles = expand_ellipsis(row_styles, rnlevels)
  color_indices = []
  for i in range(cnlevels):
    if column_styles[i] == "marker-color":
      color_indices.append(i)
  for i in range(rnlevels):
    if row_styles[i] == "marker-color":
      color_indices.append(cnlevels + i)
  if data_styles == "marker-color":
      color_indices.append(cnlevels + rnlevels)
  color_indices = np.asarray(color_indices)
  html = HTMLFormatter(data, rheight="40px", pwidth="100px", plot_type="bar", color_indices=color_indices, colormap=cmap).render()
  print(html)
  return
  for i in range(data.index.shape[0]):
    print("<tr>")
    draw_row(data.index[i])
    for j in range(data.columns.shape[0]):
      if is_multi_c:
        column_code = [c[j] for c in data.columns.codes]
      else:
        column_code = [j]
      if is_multi_r:
        row_code = [c[i] for c in data.index.codes]
      else:
        row_code = [i]
      column_code = np.asarray(column_code)
      row_code = np.asarray(row_code)
      color = np.hstack((column_code, row_code))[color_indices]
      color += np.ones((2,), dtype=np.uint8)
      draw_cell(data.iloc[i, j], i, j, colors2d[colors][4][color[0], color[1]])
    print("</tr>")

def draw_row(c):
  print("<th>{}</th>".format(c))

def draw_column(c):
  print("<th>{}</th>".format(c[-1]))

def draw_cell(d, i, j, color):
  td = "<td><div style='vertical-align:middle; display:inline-block; text-align:right;margin-right: 1em; line-height: 40px; height:40px; width: 4em'>{:.2f}</div><div style='vertical-align:middle; display:inline-block; width: {}px; height: 40px; background-color: {}'></div></td>".format(d, 30*d, palette.stringify_hex(color))
  print(td)
