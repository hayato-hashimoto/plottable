from collections import defaultdict, OrderedDict
from copy import copy
from enum import Enum

import numpy as np
import pandas as pd
from bottle import SimpleTemplate

from . import element_formatters
from . import miniplot

class constants(Enum):
  linebreak = 0
  merge = 1
  leaf = 2

def calc_span(c, base_span=1):
  if isinstance(c, pd.MultiIndex):
    ret = np.zeros((c.nlevels, len(c)))
    higher=False
    for i, code in enumerate(c.codes):
      code = np.asarray(code)
      cond = np.concatenate([[True], code[1:] != code[:-1]])
      cond = higher | cond
      ind, = np.where(cond)
      spans = np.diff(np.concatenate([ind, [len(code)]]))
      ret[i, ind] = spans
      higher = cond
  else:
    ret = np.ones((len(c), 1))
  return base_span * ret

def calc_repeat_span(ds, cols):
  totalcols = 1
  for c in cols:
    totalcols *= len(ds.coords[c])
  repeat = 1
  spansize = []
  repeatcount = []
  for c in cols:
    L =  len(ds.coords[c])
    repeatcount.append(repeat)
    spansize.append(int(totalcols / repeat / L))
    repeat *= L
  return repeatcount, spansize

def get_keys(col, ck, row, rk):
  cdic = {k: v for k, v in zip(col, ck)}
  rdic = {k: v for k, v in zip(row, rk)}
  return {**cdic, **rdic}

def make_keys(ds, cols, coords={}):
  coords=[ds.coords[c] if not c in coords else coords[c] for c in cols]
  keys=np.meshgrid(*coords,indexing="ij")
  return [x for x in zip(*[x.flatten() for x in keys])]

template = """
% cols = ds.indexes["column"]
% rows = ds.indexes["row"]
  <table>
  <thead>
% for i, c in enumerate(cols.codes):
    <tr>
%   if i == 0:
      <td colspan="{{rows.nlevels}}" rowspan="{{cols.nlevels}}"></td>
%   end
%   for j, cc in enumerate(c):
%     if cspan[i, j] > 0:
%       style = get_col_style(i, j)
%       ck = cols.levels[i][cc]
        <th colspan="{{ cspan[i, j] }}">{{! write_object(ck, style) }}</th>
%     end
%   end
    </tr>
% end
  </thead>
  <tbody>
% for j, rt in enumerate(rows):
    <tr>
%   for i, rk in enumerate(rt):
%     if rspan[i, j] > 0:
%       style = get_row_style(i, j)
        <th rowspan="{{ rspan[i, j] }}">{{! write_object(rk, style) }}</th>
%     end
%   end
%   for i, ct in enumerate(cols):
%     if cell_mode == "minitable":
        <td class="datacell" id="{{generate_id(rt, ct)}}">
        <table class="minitable">
        <tbody>
%       for i in range(minitable_nx):
          <tr>
%         for j in range(minitable_ny):
%           idx = minitable_nx * j + i
%           if idx >= len(cellkey):
              <td></td>
%           else:
%             cellk = cellkey[idx]
%             obj = get_ds_cell(ds, rows, rk, cols, ck, cellk)
%             keys = get_keys(rows, rk, cols, ck)
%             style = get_style(cellk, obj, keys)
%             id = generate_id(rk, ck, cellk)
              <td style="{{style.get_td_css()}}" class="{{style.get_classes()}}" id="{{id}}">
              {{! write_object(obj, style) }}
              </td>
%           end
%         end
          </tr>
%       end
        </tbody>
        </table>
        </td>
%     elif cell_mode == "td":
%       for cellk in ds.data_vars:
%         obj = ds[cellk].isel({"column": i, "row": j}).data
%         style = get_style(cellk, obj, rt, ct)
%         id = generate_id(rt, ct, cellk)
          <td style="{{style.get_td_css()}}" class="{{style.get_classes()}}" id="{{id}}">
          {{! write_object(obj, style) }}
          </td>
%       end
%     elif cell_mode == "inline":
        <td class="datacell" id="{{generate_id(rows, rk, cols, ck)}}">
%       for cellk in ds.data_vars:
%         obj = get_ds_cell(ds, rows, rk, cols, ck, cellk)
%         keys = get_keys(rows, rk, cols, ck)
%         style = get_style(cellk, obj, keys)
%         id = generate_id(rk, ck, cellk)
          <span style="{{style.get_span_css()}}" class="{{style.get_classes()}}" id="{{id}}">
          {{! write_object(obj, style) }}
          </td>
%       end
%     end
      </td>
%   end
  </tr>
% end
  </tbody>
  </table>
"""

def write_object(obj, options):
    return options.format(obj)

def default_format(x):
  if isinstance(x, (float, np.float32, np.float64)):
    return element_formatters.engineering_notation_tex(x)
  else:
    return str(x)

class Style:
  def __init__(self, **kwargs):
    pass
  def format(self, value):
    return default_format(value)
  def get_td_css(self):
    return ""
  def get_span_css(self):
    return ""
  def get_classes(self):
    return ""

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

def get_row_style(*args):
  return Style()
def get_col_style(*args):
  return Style()
def get_style(*args):
  return Style()
def generate_id(*args):
  return "_"

def variables_parser(ds, specifiers):
  variables = []
  def _loop(specifier, constraint, preceding_constraint):
    if isinstance(specifier, (list, tuple)):
      new_constraint = specifier[0]
      rest = specifier[1:]
      if not isinstance(new_constraint, (dict, OrderedDict)):
        raise TypeError("Variable specifier must be either a variable name (string) or a (list or tuple) with a constraint followed by sub-specifiers. Constraint must be a dict or an OrderedDict, not {}.".format(type(new_constraint)))
      nc_list = list(new_constraint.items())
      tmp_constraint = OrderedDict(constraint)
      for i, (dim_name, indexer) in enumerate(nc_list):
        if indexer == Ellipsis:
          for key in ds.indexes[dim_name]:
            if key in preceding_constraint[dim_name]:
              continue
            tmp_constraint[dim_name] = key
            return _loop((OrderedDict(nc_list[i+1:]), *rest), tmp_constraint, preceding_constraint)
        else:
          tmp_constraint[dim_name] = indexer
          preceding_constraint[dim_name].append(indexer)
      tmp_preceding_constraint = copy(preceding_constraint)
      for s in rest:
        _loop(s, tmp_constraint, tmp_preceding_constraint)
    else:
      # variable name
      assert isinstance(specifier, str)
      return variables.append((constraint, specifier))
  _loop(({}, *specifiers), OrderedDict(), defaultdict(list))
  return variables

class DataPlot:
  def __init__(self, name, plot_type, dataset, plot_variable, axis=None, axis_style=None):
    self.name = name
    self.plot_type = plot_type
    self.dataset = dataset
    self.variable = variable
    self.axis = axis
    self.axis_style = axis_style
    self.dims = tuple(set(variable.dims) - set([axis]))
    if self.axis is None:
      self.span_axes = []
    else:
      self.span_axes = [axis]
  def render(self): 
    return miniplot.miniplot(self.plot_type, self.dataset, axis=self.axis).sel(self.constraint)[self.variable]

class TableVariable:
  # DEBUG
  def __repr__(self):
    return "<TableVariable {}>".format(self.name)
  def __init__(self, name, dataset, variable, constraint, merge_cells=[]):
    self.name = name
    self.dataset = dataset
    self.variable = variable
    merge_cells = [k for k, v in constraint.items() if v == constants.merge]
    constraint = {k: v for k, v in constraint.items() if not v == constants.merge}
    self.constraint = constraint
    self.dims = dataset.sel(constraint)[variable].dims
    self.merge_cells = merge_cells
    self.merge_cols = []
    self.merge_rows = []

class OrderedPrefixTree:
  def __init__(self):
    self.tree = []
  def append(self, key, value):
    node = self.tree
    for k in key + [constants.leaf]:
      if node and node[-1][0] == k:
        node = node[-1][1]
      else:
        new_node = (k, [])
        node.append(new_node)
        node = new_node[1]
    node.append(value)
  def __iter__(self):
    def loop(node):
      for k, v in node:
        if k is constants.leaf:
          yield v
        else:
          yield from loop(v)
    return loop(self.tree)

def is_horizontal_block(v):
  return isinstance(v, TableVariable) and v.merge_cols

def is_vertical_block(v):
  return isinstance(v, TableVariable) and v.merge_rows

class DataTableCellTemplate:
  def __init__(self, children):
    self.children = children
    self.layout()
  # DEBUG
  def __repr__(self):
    return "<CellTemplate shape={} vars={}>".format(self.shape, [x.name for x in self.children])
  def layout(self):
    anchor_left, anchor_right, anchor_top, anchor_bottom = (0, 0, 0, 0)
    cursor_x, cursor_y = (0, 0)
    width, height = (0, 0)
    layouts = []
    tmp_layouts = []
    for v in self.children:
      if is_horizontal_block(v):
        for t in tmp_layouts:
          layouts.append((t[0], slice(t[1], anchor_bottom), t[2]))
        layouts.append((slice(None), anchor_bottom, v))
        cursor_x = 0
        cursor_y = anchor_bottom + 1
        width = max(1, width)
        height = max(height, cursor_y)
        anchor_left = 0
        anchor_right = 0
        anchor_top = cursor_y
        anchor_bottom = cursor_y
        tmp_layouts = []
      elif is_vertical_block(v):
        tmp_layouts.append((anchor_right, anchor_top, v))
        cursor_x = anchor_right + 1
        cursor_y = anchor_top
        anchor_bottom = max(anchor_bottom, cursor_y + 1)
        width = max(width, cursor_x)
        height = max(height, anchor_bottom)
        anchor_left = cursor_x
        anchor_right = cursor_x
      elif v == constants.linebreak:
        cursor_x = anchor_left
        cursor_y += 1
        anchor_bottom = max(anchor_bottom, cursor_y)
        height = max(height, cursor_y)
      else:
        # first element in the line
        if cursor_x == anchor_left:
          anchor_bottom = max(anchor_bottom, cursor_y + 1)
          height = max(height, anchor_bottom)
        layouts.append((cursor_x, cursor_y, v))
        cursor_x += 1
        anchor_right = max(anchor_right, cursor_x)
        width = max(width, cursor_x)
    for t in tmp_layouts:
      layouts.append((t[0], slice(t[1], -1), t[2]))
    self.shape = (width, height)
    self.ij = np.zeros(self.shape, dtype=object)
    self.ij[:] = None
    self.layouts = layouts
    for x, y, v in layouts:
      self.ij[x,y] = v
  def reflow(self, width=None, height=None):
    if width is None:
      width = self.shape[0]
    if height is None:
      height = self.shape[1]
    self.shape = (width, height)
    self.ij = np.zeros(self.shape, dtype=object)
    for x, y, v in self.layouts:
      self.ij[x,y] = v

class DataTable:
  def __init__(self, dataset, variables, cols, rows, colors="Yl-BrGnBu", cellspacing=0.3, row_styles=[{}, ...], column_styles=[{"next-level": "color"}, ...], rest_styles={}, variable_styles=[{}, ...], data_styles=None, expand_variables="column"):
    self.cols = cols
    self.rows = rows
    self.variables = variables
    self.dataset = dataset
    cnlevels = len(cols)
    rnlevels = len(rows)
    nvariables = len(variables)
    self.column_styles = expand_ellipsis(column_styles, cnlevels)
    self.row_styles = expand_ellipsis(row_styles, rnlevels)
    self.variable_styles = expand_ellipsis(variable_styles, nvariables)
    self.depth_indices = defaultdict(list)
    for i in range(cnlevels):
      if "next-level" in self.column_styles[i]:
        keys = self.column_styles[i]["next-level"]
        if isinstance(keys, str):
          keys = list(keys)
        for k in keys:
          self.depth_indices[k].append(i)
    for i in range(rnlevels):
      if "next-level" in self.row_styles[i]:
        keys = self.row_styles[i]["next-level"]
        if isinstance(keys, str):
          keys = list(keys)
        for k in keys:
          depth_indices[k].append(cnlevels + i)

    column_index_tree = OrderedPrefixTree()
    for v in variables:
      if v == constants.linebreak:
        if dimspec is None:
          raise ValueError("DataTable variables must not start with linebreak")
      else:
        dimspec = [c in v.dims for c in cols]
      if v.merge_cells:
        for i in range(len(dimspec)):
          if dimspec[-i-1] or cols[-i-1] not in v.merge_cells:
            break
          v.merge_cols = cols[-i-1]
          dimspec[-i-1] = True
      column_index_tree.append(dimspec, v)

    row_index_tree = OrderedPrefixTree()
    dimspec = None
    for v in variables:
      if v == constants.linebreak:
        if dimspec is None:
          raise ValueError("DataTable variables must not start with linebreak")
      else:
        dimspec = [c in v.dims for c in rows]
      if v.merge_cells:
        for i in range(len(dimspec)):
          if dimspec[-i-1] or rows[i] not in v.merge_cells:
            break
          v.merge_rows = rows[-i-1]
          dimspec[-i-1] = True
      row_index_tree.append(dimspec, v)

    def intersection(a, b):
      return [x for x in a if x in b]

    # prepare cell templates
    for c in column_index_tree:
      templates = [DataTableCellTemplate(intersection(c, r)) for r in row_index_tree]
      w = max([x.shape[0] for x in templates])
      for x in templates:
        x.reflow(width=w)
      # TODO:append?
      c.append(templates)

    # prepare column codes
    code = [None] * len(self.cols)
    codes = []
    def loop(depth, nodes):
      for node in nodes:
        # leaf node
        if node[0] is constants.leaf:
          nodes = node[1]
          w = max([x.shape[0] for x in nodes[-1]])
          for idx in range(w):
            codes.append(code + [idx])
        # True node (vector dimension)
        elif node[0]:
          for c in range(len(self.dataset.indexes[cols[depth]])):
            code[depth] = c
            loop(depth+1, node[1])
        # False node (scalar dimension)
        else:
          code[depth] = None
          loop(depth+1, node[1])
    loop(0, column_index_tree.tree)
    print(str(column_index_tree.tree))
    self.column_codes = codes

  @staticmethod
  def from_dataset(dataset, cols, rows, variables=None, **kwargs):
    dims = dataset.dims
    table_variables = []
    if variables is None:
      variables = list(dataset.data_vars)
    _variables = variables_parser(dataset, variables)
    # TODO expand styles
    #data = data.stack({"column": cols, "row": rows})
    #ret = data._replace(variables={"column": data.column, "row": data.row})
    for constraint, variable in _variables:
      if len(constraint) > 0:
        name = (tuple(constraint.items()), variable)
      else:
        name = variable
      if variable.split(".")[-1] in miniplot.types:
        rest = list(set(dims) - set(cols) - set(rows))
        axis = rest[0] if len(rest) > 0 else None
        plot_type = variable.split(".")[-1]
        plot_variable = "".join(variable.split(".")[:-1])
        table_variables.append(DataPlot(name, dataset, plot_variable, constraint, axis))
      else:
        table_variables.append(TableVariable(name, dataset, variable, constraint))
    return DataTable(dataset, table_variables, cols, rows, **kwargs)

  def to_html(self, plot_format="html"):
    temp = SimpleTemplate(template)
    return temp.render(datatable=self,get_style=get_style,get_col_style=get_col_style,get_row_style=get_row_style,generate_id=generate_id,write_object=write_object)

  def _get_hex_color(self, name):
    self.colormap[name].get(self.style_indexer[name])

  def _get_level(name, dataset):
    a = []
    for s in style_levels_column[name]:
       ls = dataset.indexes["column"].codes[s]
       if not use_categorical_style_indexer[name][level]:
         ls = dataset.indexes.levels[ls]
