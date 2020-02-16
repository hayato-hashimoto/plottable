from collections import OrderedDict

import numpy as np

from pandas.core.dtypes.generic import ABCMultiIndex
from pandas import option_context
from pandas.io.common import _is_url
from pandas.io.formats.format import DataFrameFormatter, get_level_lengths
from pandas.io.formats.printing import pprint_thing

from . import element_formatters

def defaults_or_kwargs(defaults, kwargs):
  ret = { **defaults }
  for k in defaults:
    if k in kwargs:
      ret[k] = kwargs.pop(k)
  return ret

class HTMLFormatter(DataFrameFormatter):
    """
    Copied from pandas/io/html.py 
    """

    indent_delta = 2

    def __init__(self, *args, **kwargs):
        defaults = {
            "rheight":"30px",
            "pwidth":"200px",
            "ppos":"right",
            "plot_type":"bar",
            "color_indices":None,
            "colormap":None,
            "classes":None,
            "border":None }
        html_kwargs = defaults_or_kwargs(defaults, kwargs)
        self.rheight = html_kwargs["rheight"]
        self.pwidth = html_kwargs["pwidth"]
        self.ppos = html_kwargs["ppos"]
        self.plot_type = html_kwargs["plot_type"]
        self.color_indices = html_kwargs["color_indices"]
        self.colormap = html_kwargs["colormap"]
        self.classes = html_kwargs["classes"]
        self.border = html_kwargs["border"]
        kwargs["float_format"] = element_formatters.engineering_notation_ascii
        super(HTMLFormatter, self).__init__(*args, **kwargs)
        self.columns = self.tr_frame.columns
        self.is_multi_c = isinstance(self.tr_frame.columns, ABCMultiIndex)
        self.is_multi_r = isinstance(self.tr_frame.index, ABCMultiIndex)
        self.elements = []
        self.escape = self.kwds.get('escape', True)
        #if self.border is None:
        #    self.border = False #get_option('display.html.border')
        if isinstance(self.col_space, int):
            self.col_space = ('{colspace}px'
                                  .format(colspace=self.col_space))
    @property
    def row_levels(self):
        if self.index:
            # showing (row) index
            return self.tr_frame.index.nlevels
        elif self.show_col_idx_names:
            # see gh-22579
            # Column misalignment also occurs for
            # a standard index when the columns index is named.
            # If the row index is not displayed a column of
            # blank cells need to be included before the DataFrame values.
            return 1
        # not showing (row) index
        return 0

    def _get_columns_formatted_values(self):
        return self.columns

    @property
    def ncols(self):
        return len(self.tr_frame.columns)

    def write(self, s, indent=0):
        rs = pprint_thing(s)
        self.elements.append(' ' * indent + rs)

    def write_th(self, s, header=False, indent=0, tags=None):
        """
        Method for writting a formatted <th> cell.
        If col_space is set on the formatter then that is used for
        the value of min-width.
        Parameters
        ----------
        s : object
            The data to be written inside the cell.
        header : boolean, default False
            Set to True if the <th> is for use inside <thead>.  This will
            cause min-width to be set if there is one.
        indent : int, default 0
            The indentation level of the cell.
        tags : string, default None
            Tags to include in the cell.
        Returns
        -------
        A written <th> cell.
        """
        if header and self.col_space is not None:
            tags = (tags or "")
            tags += ('style="min-width: {colspace};"'
                     .format(colspace=self.col_space))

        return self._write_cell(s, kind='th', indent=indent, tags=tags)

    def write_td(self, s, indent=0, tags=None, i=None, j=None):
        return self._write_cell(s, kind='td', indent=indent, tags=tags, i=i, j=j)

    def miniplot(self, s, h, w, color, plot_type='bar', scale="area"):
        if plot_type == 'bar':
          rs = '<div style="display:inline-block;min-width:{w}"><div style="display:inline-block;height:{h};line-height:{h};vertical-align:middle;width:{barw};background-color:{c}"></div></div>'.format(s=s, h=h, w=w, barw=20*s, c=color)
        elif plot_type == 'square':
             if scale == "area":
               a = math.sqrt(s)
             elif scale == "linear":
               a = s
             else:
               raise ValueError("scale must be 'area' or 'linear'")
             rs = '<svg x="0px" y="0px" width="{w}px" height="{h}px" viewBox="-1 -1 1 1"><rect fill="{c}" x="{x}" y="{y}" width="{rw}" height="{rh}"/><rect stroke="{c}"/></svg>'.format(w=w, h=h, c=color, x=-a, y=-a, rw=2*a, rh=2*a)
        return rs

    def _write_cell(self, s, kind='td', indent=0, tags=None, i=None, j=None):
        if self.plot_type is not None and i is not None and j is not None:
            if self.is_multi_c:
              column_code = [c[j] for c in self.tr_frame.columns.codes]
            else:
              column_code = [j]
            if self.is_multi_r:
              row_code = [c[i] for c in self.tr_frame.index.codes]
            else:
              row_code = [i]
            column_code = np.asarray(column_code)
            row_code = np.asarray(row_code)
            color = np.hstack((column_code, row_code))[self.color_indices]
            color += np.ones((2,), dtype=np.uint8)
            color = self.colormap(color[0], color[1])
            miniplot_str = self.miniplot(self.tr_frame.iloc[i, j], self.rheight, self.pwidth, color, self.plot_type)

        if tags is not None:
            start_tag = '<{kind} {tags}>'.format(kind=kind, tags=tags)
        else:
            start_tag = '<{kind}>'.format(kind=kind)
            tags = ""

        start_miniplot_tag = '<{kind} {tags}>'.format(kind=kind, tags=tags+' style="padding-right: 1em;"')

        if self.escape:
            # escape & first to prevent double escaping of &
            esc = OrderedDict([('&', r'&amp;'), ('<', r'&lt;'),
                               ('>', r'&gt;')])
        else:
            esc = {}

        rs = s

        if self.render_links and _is_url(rs):
            rs_unescaped = pprint_thing(s, escape_chars={}).strip()
            rs = '<a href="{url}" target="_blank">{rs}</a>'.format(url=rs_unescaped, rs=rs)

        if self.plot_type is not None and i is not None and j is not None and self.ppos == "left":
                self.write('{start}{rs}</{kind}>'.format(
          start=start_miniplot_tag, rs=miniplot_str, kind=kind), indent)
        self.write('{start}{rs}</{kind}>'.format(
          start=start_tag, rs=rs, kind=kind), indent)
        if self.plot_type is not None and i is not None and j is not None and self.ppos == "right":
                self.write('{start}{rs}</{kind}>'.format(
          start=start_miniplot_tag, rs=miniplot_str, kind=kind), indent)

    def write_tr(self, line, indent=0, indent_delta=0, header=False,
                 align=None, tags=None, nindex_levels=0, i=0):
        if tags is None:
            tags = {}

        if align is None:
            self.write('<tr>', indent)
        else:
            self.write('<tr style="text-align: {align};">'
                       .format(align=align), indent)
        indent += indent_delta

        for j, s in enumerate(line):
            val_tag = tags.get(j, None)
            if header or j < nindex_levels:
                self.write_th(s, indent=indent, header=header, tags=val_tag)
            else:
                self.write_td(s, indent, tags=val_tag, i=i, j=j-nindex_levels)

        indent -= indent_delta
        self.write('</tr>', indent)

    def render(self):
        self._write_table()

        if self.should_show_dimensions:
            by = chr(215)  # ×
            self.write('<p>{rows} rows {by} {cols} columns</p>'
                       .format(rows=len(self.frame),
                               by=by,
                               cols=len(self.frame.columns)))

        return "\n".join(self.elements)

    def _write_table(self, indent=0):
        _classes = ['dataframe']  # Default class.
        use_mathjax = True # get_option("display.html.use_mathjax")
        if not use_mathjax:
            _classes.append('tex2jax_ignore')
        if self.classes is not None:
            if isinstance(self.classes, str):
                self.classes = self.classes.split()
            if not isinstance(self.classes, (list, tuple)):
                raise TypeError('classes must be a string, list, or tuple, '
                                'not {typ}'.format(typ=type(self.classes)))
            _classes.extend(self.classes)

        if self.table_id is None:
            id_section = ""
        else:
            id_section = ' id="{table_id}"'.format(table_id=self.table_id)

        self.write('<table class="{cls}"{id_section}>'
                   .format(border=self.border, cls=' '.join(_classes),
                           id_section=id_section), indent)

        if self.header or self.show_row_idx_names:
            self._write_header(indent + self.indent_delta)

        self._write_body(indent + self.indent_delta)

        self.write('</table>', indent)

    def _write_col_header(self, indent, span_per_cell=1):
        truncate_h = self.truncate_h
        if isinstance(self.columns, ABCMultiIndex):
            template = 'colspan="{span:d}" halign="left"'

            if self.sparsify:
                # GH3547
                sentinel = object()
            else:
                sentinel = False
            levels = self.columns.format(sparsify=sentinel, adjoin=False,
                                         names=False)
            level_lengths = get_level_lengths(levels, sentinel)
            inner_lvl = len(level_lengths) - 1
            for lnum, (records, values) in enumerate(zip(level_lengths,
                                                         levels)):
                if truncate_h:
                    # modify the header lines
                    ins_col = self.tr_col_num
                    if self.sparsify:
                        recs_new = {}
                        # Increment tags after ... col.
                        for tag, span in list(records.items()):
                            if tag >= ins_col:
                                recs_new[tag + 1] = span
                            elif tag + span > ins_col:
                                recs_new[tag] = span + 1
                                if lnum == inner_lvl:
                                    values = (values[:ins_col] + ('...',) +
                                              values[ins_col:])
                                else:
                                    # sparse col headers do not receive a ...
                                    values = (values[:ins_col] +
                                              (values[ins_col - 1], ) +
                                              values[ins_col:])
                            else:
                                recs_new[tag] = span
                            # if ins_col lies between tags, all col headers
                            # get ...
                            if tag + span == ins_col:
                                recs_new[ins_col] = 1
                                values = (values[:ins_col] + ('...',) +
                                          values[ins_col:])
                        records = recs_new
                        inner_lvl = len(level_lengths) - 1
                        if lnum == inner_lvl:
                            records[ins_col] = 1
                    else:
                        recs_new = {}
                        for tag, span in list(records.items()):
                            if tag >= ins_col:
                                recs_new[tag + 1] = span
                            else:
                                recs_new[tag] = span
                        recs_new[ins_col] = 1
                        records = recs_new
                        values = (values[:ins_col] + ['...'] +
                                  values[ins_col:])

                # see gh-22579
                # Column Offset Bug with to_html(index=False) with
                # MultiIndex Columns and Index.
                # Initially fill row with blank cells before column names.
                # TODO: Refactor to remove code duplication with code
                # block below for standard columns index.
                row = [''] * (self.row_levels - 1)
                if self.index or self.show_col_idx_names:
                    # see gh-22747
                    # If to_html(index_names=False) do not show columns
                    # index names.
                    # TODO: Refactor to use _get_column_name_list from
                    # DataFrameFormatter class and create a
                    # _get_formatted_column_labels function for code
                    # parity with DataFrameFormatter class.
                    if self.show_index_names:
                        name = self.columns.names[lnum]
                        row.append(pprint_thing(name or ''))
                    else:
                        row.append('')

                tags = {}
                j = len(row)
                for i, v in enumerate(values):
                    if i in records:
                        if span_per_cell * records[i] > 1:
                            tags[j] = template.format(span=span_per_cell * records[i])
                    else:
                        continue
                    j += 1
                    row.append(v)
                self.write_tr(row, indent, self.indent_delta, tags=tags,
                              header=True)
        else:
            # see gh-22579
            # Column misalignment also occurs for
            # a standard index when the columns index is named.
            # Initially fill row with blank cells before column names.
            # TODO: Refactor to remove code duplication with code block
            # above for columns MultiIndex.
            row = [''] * (self.row_levels - 1)
            if self.index or self.show_col_idx_names:
                # see gh-22747
                # If to_html(index_names=False) do not show columns
                # index names.
                # TODO: Refactor to use _get_column_name_list from
                # DataFrameFormatter class.
                if self.show_index_names:
                    row.append(self.columns.name or '')
                else:
                    row.append('')
            row.extend(self._get_columns_formatted_values())
            align = self.justify

            if truncate_h:
                ins_col = self.row_levels + self.tr_col_num
                row.insert(ins_col, '...')

            self.write_tr(row, indent, self.indent_delta, header=True,
                          align=align)

    def _write_row_header(self, indent):
        truncate_h = self.truncate_h
        row = ([x if x is not None else '' for x in self.tr_frame.index.names]
               + [''] * (self.ncols + (1 if truncate_h else 0)))
        self.write_tr(row, indent, self.indent_delta, header=True)

    def _write_header(self, indent):
        self.write('<thead>', indent)

        if self.header:
            self._write_col_header(indent + self.indent_delta, 2)

        if self.show_row_idx_names:
            self._write_row_header(indent + self.indent_delta)

        self.write('</thead>', indent)

    def _get_formatted_values(self):
        with option_context('display.max_colwidth', 999999):
            fmt_values = {i: self._format_col(i)
                          for i in range(self.ncols)}
        return fmt_values

    def _write_body(self, indent):
        self.write('<tbody>', indent)
        fmt_values = self._get_formatted_values()

        # write values
        if self.index and isinstance(self.tr_frame.index, ABCMultiIndex):
            self._write_hierarchical_rows(
                fmt_values, indent + self.indent_delta)
        else:
            self._write_regular_rows(
                fmt_values, indent + self.indent_delta)

        self.write('</tbody>', indent)

    def _write_regular_rows(self, fmt_values, indent):
        truncate_h = self.truncate_h
        truncate_v = self.truncate_v

        nrows = len(self.tr_frame)

        if self.index:
            fmt = self._get_formatter('__index__')
            if fmt is not None:
                index_values = self.tr_frame.index.map(fmt)
            else:
                index_values = self.tr_frame.index.format()

        row = []
        for i in range(nrows):

            if truncate_v and i == (self.tr_row_num):
                str_sep_row = ['...'] * len(row)
                self.write_tr(str_sep_row, indent, self.indent_delta,
                              tags=None, nindex_levels=self.row_levels)

            row = []
            if self.index:
                row.append(index_values[i])
            # see gh-22579
            # Column misalignment also occurs for
            # a standard index when the columns index is named.
            # Add blank cell before data cells.
            elif self.show_col_idx_names:
                row.append('')
            row.extend(fmt_values[j][i] for j in range(self.ncols))

            if truncate_h:
                dot_col_ix = self.tr_col_num + self.row_levels
                row.insert(dot_col_ix, '...')
            self.write_tr(row, indent, self.indent_delta, tags=None,
                          nindex_levels=self.row_levels, i=i)

    def _write_hierarchical_rows(self, fmt_values, indent):
        template = 'rowspan="{span}" valign="top"'

        truncate_h = self.truncate_h
        truncate_v = self.truncate_v
        frame = self.tr_frame
        nrows = len(frame)

        idx_values = frame.index.format(sparsify=False, adjoin=False,
                                        names=False)
        idx_values = list(zip(*idx_values))

        if self.sparsify:
            # GH3547
            sentinel = object()
            levels = frame.index.format(sparsify=sentinel, adjoin=False,
                                        names=False)

            level_lengths = get_level_lengths(levels, sentinel)
            inner_lvl = len(level_lengths) - 1
            if truncate_v:
                # Insert ... row and adjust idx_values and
                # level_lengths to take this into account.
                ins_row = self.tr_row_num
                inserted = False
                for lnum, records in enumerate(level_lengths):
                    rec_new = {}
                    for tag, span in list(records.items()):
                        if tag >= ins_row:
                            rec_new[tag + 1] = span
                        elif tag + span > ins_row:
                            rec_new[tag] = span + 1

                            # GH 14882 - Make sure insertion done once
                            if not inserted:
                                dot_row = list(idx_values[ins_row - 1])
                                dot_row[-1] = '...'
                                idx_values.insert(ins_row, tuple(dot_row))
                                inserted = True
                            else:
                                dot_row = list(idx_values[ins_row])
                                dot_row[inner_lvl - lnum] = '...'
                                idx_values[ins_row] = tuple(dot_row)
                        else:
                            rec_new[tag] = span
                        # If ins_row lies between tags, all cols idx cols
                        # receive ...
                        if tag + span == ins_row:
                            rec_new[ins_row] = 1
                            if lnum == 0:
                                idx_values.insert(ins_row, tuple(
                                    ['...'] * len(level_lengths)))

                            # GH 14882 - Place ... in correct level
                            elif inserted:
                                dot_row = list(idx_values[ins_row])
                                dot_row[inner_lvl - lnum] = '...'
                                idx_values[ins_row] = tuple(dot_row)
                    level_lengths[lnum] = rec_new

                level_lengths[inner_lvl][ins_row] = 1
                for ix_col in range(len(fmt_values)):
                    fmt_values[ix_col].insert(ins_row, '...')
                nrows += 1

            for i in range(nrows):
                row = []
                tags = {}

                sparse_offset = 0
                j = 0
                for records, v in zip(level_lengths, idx_values[i]):
                    if i in records:
                        if records[i] > 1:
                            tags[j] = template.format(span=records[i])
                    else:
                        sparse_offset += 1
                        continue

                    j += 1
                    row.append(v)

                row.extend(fmt_values[j][i] for j in range(self.ncols))
                if truncate_h:
                    row.insert(self.row_levels - sparse_offset +
                               self.tr_col_num, '...')
                self.write_tr(row, indent, self.indent_delta, tags=tags,
                              nindex_levels=len(levels) - sparse_offset)
        else:
            row = []
            for i in range(len(frame)):
                if truncate_v and i == (self.tr_row_num):
                    str_sep_row = ['...'] * len(row)
                    self.write_tr(str_sep_row, indent, self.indent_delta,
                                  tags=None, nindex_levels=self.row_levels)

                idx_values = list(zip(*frame.index.format(
                    sparsify=False, adjoin=False, names=False)))
                row = []
                row.extend(idx_values[i])
                row.extend(fmt_values[j][i] for j in range(self.ncols))
                if truncate_h:
                    row.insert(self.row_levels + self.tr_col_num, '...')
                self.write_tr(row, indent, self.indent_delta, tags=None,
                              nindex_levels=frame.index.nlevels)
