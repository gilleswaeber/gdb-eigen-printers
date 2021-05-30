# -*- coding: utf-8 -*-
# This file is part of Eigen, a lightweight C++ template library
# for linear algebra.
#
# Copyright (C) 2021 Gilles Waeber <moi@gilleswaeber.ch>
# Copyright (C) 2009 Benjamin Schindler <bschindler@inf.ethz.ch>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Pretty printers for Eigen::Matrix
# This is still pretty basic as the python extension to gdb is still pretty basic. 
# It cannot handle complex eigen types and it doesn't support many of the other eigen types
# This code supports fixed size as well as dynamic size matrices

# To use it:
#
# * Create a directory and put the file as well as an empty __init__.py in 
#   that directory.
# * Create a ~/.gdbinit file, that contains the following:
#      python
#      import sys
#      sys.path.insert(0, '/path/to/eigen/printer/directory')
#      from gdb_eigen_printers import register_eigen_printers
#      register_eigen_printers(None)
#      end

import re
import traceback

import gdb
from itertools import chain

from compile import SCRIPT_DIR

# Here is the ugly trick…
VIRTUAL_MAT_ADDRESSES = {}
NEXT_ADDRESS = 1
MAT_ADDRESSES = {}

BY_ROW_COL_LIMIT = 500  # when there is more than XXX cols or rows, the ByRow and ByCol views will be disabled
INVALID_ROW_COL_LIMIT = 200000  # when there is more than XXX cols or rows, consider the object invalid

OBJECT_FILE = SCRIPT_DIR / 'eigenprinters.o'


class GdbTypes:
    inst = None

    def __init__(self):
        self.row_first_type = gdb.lookup_type(f'EigenPrinters::RowFirst')
        self.row_first_p = self.row_first_type.pointer()
        self.col_first_type = gdb.lookup_type(f'EigenPrinters::ColFirst')
        self.col_first_p = self.col_first_type.pointer()
        self.by_row_type = gdb.lookup_type(f'EigenPrinters::ByRow')
        self.by_row_p = self.by_row_type.pointer()
        self.by_col_type = gdb.lookup_type(f'EigenPrinters::ByCol')
        self.by_col_p = self.by_col_type.pointer()
        self.info_type = gdb.lookup_type(f'EigenPrinters::Info')
        self.info_p = self.by_col_type.pointer()

    def row_first(self, val):
        return val.address.cast(self.row_first_p).dereference().cast(self.row_first_type)

    def col_first(self, val):
        return val.address.cast(self.col_first_p).dereference().cast(self.col_first_type)

    def by_row(self, virtual_addr, rows):
        return gdb.Value(virtual_addr).cast(self.by_row_p).dereference().cast(self.by_row_type.array(rows - 1))

    def by_col(self, virtual_addr, cols):
        return gdb.Value(virtual_addr).cast(self.by_col_p).dereference().cast(self.by_col_type.array(cols - 1))

    @classmethod
    def get(cls) -> 'GdbTypes':
        if cls.inst is None:
            try:
                cls.inst = GdbTypes()
            except gdb.error as _:
                if OBJECT_FILE.is_file():
                    gdb.execute(f'add-symbol-file {OBJECT_FILE} 0')
                    cls.inst = GdbTypes()
                else:
                    print('Eigenprinter types missing and library file not found, please run compile.py')
                    raise
        return cls.inst


def unwrap(val):
    if val.type.code == gdb.TYPE_CODE_INT:
        return int(val)
    elif val.type.code == gdb.TYPE_CODE_FLT:
        return float(val)
    elif 'CompressedStorage' in val.type.name:
        act_type = re.sub('^.*CompressedStorage<', '', val.type.name).split(',', 1)[0]
        if act_type in ('double', 'float'):
            return float(val)
        elif 'int' in act_type or 'long' in act_type:
            return int(val)
        else:
            raise Exception(f'Cannot unwrap compressed {val.type}')
    else:
        raise Exception(f'Cannot unwrap {val.type}')


def format_val(val):
    v = unwrap(val)
    if isinstance(v, int):
        return str(v)
    elif isinstance(v, float):
        return f'{v:.4g}'
    else:
        return str(v)


class MatrixStringifier:
    def __init__(self, variety, val):
        global MAT_ADDRESSES
        self.address = int(val.address)
        self.variety = variety
        if self.address in MAT_ADDRESSES:
            self.found = True
            self.matrix = MAT_ADDRESSES[self.address]
        else:
            self.found = False
        pass

    def display_hint(self):
        if self.found:
            return 'array'
        else:
            return 'string'

    @staticmethod
    def cell_data(rcv):
        row, col, value = rcv
        return f'[{row},{col}]', value

    def children(self):
        if self.found:
            if self.variety == 'RowFirst':
                return map(self.cell_data, self.matrix.row_first_iterator())
            elif self.variety == 'ColFirst':
                return map(self.cell_data, self.matrix.col_first_iterator())
            else:
                return iter([('error', self.to_string())])
        else:
            return iter([('error', self.to_string())])

    def to_string(self):
        if self.found:
            return self.variety
        else:
            return f'Error: matrix not found {self.address}'


class MatrixPartStringifier:
    def __init__(self, variety, val):
        global VIRTUAL_MAT_ADDRESSES
        self.virt_address = int(val.address)
        self.variety = variety
        if self.virt_address in VIRTUAL_MAT_ADDRESSES:
            self.found = True
            self.matrix, self.i = VIRTUAL_MAT_ADDRESSES[self.virt_address]
        else:
            self.found = False

    def display_hint(self):
        return 'string'

    def to_string(self):
        if self.found:
            if self.variety == 'ByRow':
                vals = {k: v for k, v in self.matrix.row_iterator(self.i)}
                return ' '.join(
                    f'{format_val(vals[k]):>7}' if k in vals else '      0' for k in range(self.matrix.cols))
            elif self.variety == 'ByCol':
                vals = {k: v for k, v in self.matrix.col_iterator(self.i)}
                return ' '.join(
                    f'{format_val(vals[k]):>7}' if k in vals else '      0' for k in range(self.matrix.rows))
            else:
                return f'Unknown part type {self.variety}'
        else:
            return f'Error: virtual matrix not found at {self.virt_address}'


class EigenMatrix:
    """Provide iterators over a Eigen matrix/vector/…"""

    def __init__(self, val: gdb.Value):
        # The gdb extension does not support value template arguments - need to extract them by hand
        val_type = val.type
        if val_type.code == gdb.TYPE_CODE_REF:
            val_type = val_type.target()
        self.type = val_type.unqualified().strip_typedefs()
        self.invalid = False
        tag = self.type.tag
        regex = re.compile(r'<.*>')
        m: str = regex.findall(tag)[0][1:-1]
        template_params = m.split(',')
        template_params = [x.replace(" ", "") for x in template_params]

        if template_params[1] == '-0x00000000000000001'\
                or template_params[1] == '-0x000000001'\
                or template_params[1] == '-1':
            self.rows = val['m_storage']['m_rows']
        else:
            self.rows = int(template_params[1])

        if template_params[2] == '-0x00000000000000001'\
                or template_params[2] == '-0x000000001'\
                or template_params[2] == '-1':
            self.cols = val['m_storage']['m_cols']
        else:
            self.cols = int(template_params[2])

        if self.rows < 0 or self.cols < 0:
            self.invalid = True
            return
        if self.rows > INVALID_ROW_COL_LIMIT or self.cols > INVALID_ROW_COL_LIMIT:
            self.invalid = True
            return

        self.options = 0  # default value
        if len(template_params) > 3:
            self.options = template_params[3]

        self.row_major = bool(int(self.options) & 0x1)

        self.innerType = self.type.template_argument(0)

        self.val = val

        # Fixed size matrices have a struct as their storage, so we need to walk through this
        self.data = self.val['m_storage']['m_data']
        if self.data.type.code == gdb.TYPE_CODE_STRUCT:
            self.data = self.data['array']
            self.data = self.data.cast(self.innerType.pointer())

    def col_first_iterator(self):
        for col in range(self.cols):
            for row, value in self.col_iterator(col):
                yield row, col, value

    def col_iterator(self, col):
        if col >= self.cols:
            return
        for row in range(self.rows):
            if self.row_major == 0:
                offset = col * self.rows + row
            else:
                offset = row * self.cols + col
            yield row, self.value(offset)

    def row_first_iterator(self):
        for row in range(self.rows):
            for col, value in self.row_iterator(row):
                yield row, col, value

    def row_iterator(self, row):
        if row >= self.rows:
            return
        for col in range(self.cols):
            if self.row_major == 0:
                offset = col * self.rows + row
            else:
                offset = row * self.cols + col
            yield col, self.value(offset)

    def value(self, offset):
        return (self.data + offset).dereference()


class EigenMatrixPrinter:
    """Print Eigen Matrix or Array of some kind"""

    def __init__(self, variety, val):
        # Save the variety (presumably "Matrix" or "Array") for later usage
        self.variety = variety
        self.matrix = EigenMatrix(val)

    def item_data(self, index):
        value = (self.matrix.data + index).dereference()
        return f'[{index}]', value

    @staticmethod
    def cell_data(rcv):
        row, col, value = rcv
        return f'[{row},{col}]', value

    def children(self):
        if self.matrix.invalid:
            return iter([
                ('rows', self.matrix.rows),
                ('cols', self.matrix.cols),
                ('invalid', self.matrix.invalid),
            ])
        global MAT_ADDRESSES, VIRTUAL_MAT_ADDRESSES, NEXT_ADDRESS
        variants = []
        virtual_addr = NEXT_ADDRESS
        MAT_ADDRESSES[int(self.matrix.val.address)] = self.matrix
        try:
            if self.matrix.rows > 1 and self.matrix.cols > 1:
                t = GdbTypes.get()
                variants.append(('RowFirst', t.row_first(self.matrix.val)))
                variants.append(('ColFirst', t.col_first(self.matrix.val)))
                if self.matrix.cols <= BY_ROW_COL_LIMIT and self.matrix.rows <= BY_ROW_COL_LIMIT:
                    variants.append(('ByRow', t.by_row(virtual_addr, self.matrix.rows)))
                    variants.append(('ByCol', t.by_col(virtual_addr, self.matrix.cols)))

                for i in range(max(self.matrix.rows, self.matrix.cols)):
                    VIRTUAL_MAT_ADDRESSES[NEXT_ADDRESS] = (self.matrix, i)
                    NEXT_ADDRESS += t.by_row_type.sizeof
        except Exception as e:
            traceback.print_exc()
            variants.append(('error', str(e)))
        if len(variants):
            variants.append(('debugId', virtual_addr))
        elif self.matrix.rows > 1 and self.matrix.cols > 1:
            variants = map(self.cell_data, self.matrix.row_first_iterator())
        else:
            variants = map(self.item_data, range(self.matrix.cols * self.matrix.rows))
        return chain((
            ('rows', self.matrix.rows),
            ('cols', self.matrix.cols),
            ('rowMajor', self.matrix.row_major),
        ),
            variants
        )

    def display_hint(self):
        return None

    def to_string(self):
        return "Eigen::%s<%s,%d,%d,%s> (data ptr: %s)" % (
            self.variety, self.matrix.innerType, self.matrix.rows, self.matrix.cols,
            "RowMajor" if self.matrix.row_major else "ColMajor", self.matrix.data)


class EigenSparseMatrix:
    """Provide iterators over a Eigen SparseMatrix"""

    def __init__(self, val: gdb.Value):
        val_type = val.type
        if val_type.code == gdb.TYPE_CODE_REF:
            val_type = val_type.target()
        self.type = val_type.unqualified().strip_typedefs()
        tag = self.type.tag
        regex = re.compile(r'<.*>')
        m: str = regex.findall(tag)[0][1:-1]
        template_params = m.split(',')
        template_params = [x.replace(" ", "") for x in template_params]

        self.options = 0
        if len(template_params) > 1:
            self.options = template_params[1]

        self.row_major = bool(int(self.options) & 0x1)

        self.inner_type = self.type.template_argument(0)

        self.val = val

        self.data = self.val['m_data']
        self.data = self.data.cast(self.inner_type.pointer())

        self.compressed = int(val['m_innerNonZeros']) == 0
        self.outer = int(self.val['m_outerSize'])
        self.inner = int(self.val['m_innerSize'])
        if self.row_major:
            self.rows = self.outer
            self.cols = self.inner
        else:
            self.rows = self.inner
            self.cols = self.outer

        # Count non-zeros (SparseCompressedBase.h)
        if self.compressed:
            self.non_zero = int(val['m_outerIndex'][self.outer] - val['m_outerIndex'][0])
        elif self.outer == 0:
            self.non_zero = 0
        else:
            self.non_zero = sum(int(val['m_innerNonZeros'][i]) for i in range(self.outer))

    def outer_first_iterator(self):
        for outer in range(self.outer):
            for inner, value in self.outer_iterator(outer):
                yield outer, inner, value

    def outer_iterator(self, outer):
        start = self.val['m_outerIndex'][outer]
        if self.compressed:
            end = self.val['m_outerIndex'][outer + 1]
        else:
            end = start + self.val['m_innerNonZeros'][outer]
        for index in range(int(start), int(end)):
            inner = int(self.val['m_data']['m_indices'][index])
            yield inner, self.val['m_data']['m_values'][index]

    def inner_first_iterator(self):
        its = [(outer, self.outer_iterator(outer)) for outer in range(self.outer)]
        its = [(outer, it, next(it, None)) for outer, it in its]
        for inner in range(self.inner):
            its = list(filter(lambda oic: oic[2] is not None, its))  # remove finished iterators
            for outer, it, (i, value) in its:
                if inner == i:
                    yield outer, inner, value
            its = [(outer, it, next(it, None) if i == inner else (i, value)) for outer, it, (i, value) in its]

    def inner_iterator(self, inner):
        for outer in range(self.outer):
            it = self.outer_iterator(outer)
            cur = next(it, None)
            while cur is not None and cur[0] < inner:
                cur = next(it, None)
            if cur is not None and cur[0] == inner:
                yield outer, cur[1]

    def row_first_iterator(self):
        if self.row_major:
            return self.outer_first_iterator()
        else:
            for col, row, value in self.inner_first_iterator():
                yield row, col, value

    def row_iterator(self, row):
        if self.row_major:
            return self.outer_iterator(row)
        else:
            return self.inner_iterator(row)

    def col_first_iterator(self):
        if self.row_major:
            return self.inner_first_iterator()
        else:
            for col, row, value in self.outer_first_iterator():
                yield row, col, value

    def col_iterator(self, col):
        if self.row_major:
            return self.inner_iterator(col)
        else:
            return self.outer_iterator(col)


class EigenSparseMatrixPrinter:
    """Print an Eigen SparseMatrix"""

    def __init__(self, variety, val: gdb.Value):
        self.variety = variety
        self.matrix = EigenSparseMatrix(val)

    @staticmethod
    def cell_data(rcv):
        row, col, value = rcv
        return f'[{row},{col}]', value

    def children(self):
        global NEXT_ADDRESS, MAT_ADDRESSES, VIRTUAL_MAT_ADDRESSES
        if self.matrix.data:
            variants = []
            virtual_addr = NEXT_ADDRESS
            MAT_ADDRESSES[int(self.matrix.val.address)] = self.matrix
            try:
                if self.matrix.rows > 1 and self.matrix.cols > 1:
                    t = GdbTypes.get()
                    variants.append(('RowFirst', t.row_first(self.matrix.val)))
                    variants.append(('ColFirst', t.col_first(self.matrix.val)))
                    if self.matrix.cols <= BY_ROW_COL_LIMIT and self.matrix.rows <= BY_ROW_COL_LIMIT:
                        variants.append(('ByRow', t.by_row(virtual_addr, self.matrix.rows)))
                        variants.append(('ByCol', t.by_col(virtual_addr, self.matrix.cols)))

                    for i in range(max(self.matrix.rows, self.matrix.cols)):
                        VIRTUAL_MAT_ADDRESSES[NEXT_ADDRESS] = (self.matrix, i)
                        NEXT_ADDRESS += t.by_row_type.sizeof
            except Exception as e:
                print(e)
                variants.append(('error', str(e)))
            if not len(variants):
                if self.matrix.row_major:
                    variants = map(self.cell_data, self.matrix.row_first_iterator())
                else:
                    variants = map(self.cell_data, self.matrix.col_first_iterator())
            return chain((
                ('rows', self.matrix.rows),
                ('cols', self.matrix.cols),
                ('nonZero', self.matrix.non_zero),
                ('compressed', self.matrix.compressed),
                ('rowMajor', self.matrix.row_major),
            ),
                variants)

        return iter([
            ('rows', self.matrix.rows),
            ('cols', self.matrix.cols),
            ('rowMajor', self.matrix.row_major),
            ('empty', True),
        ])  # empty matrix, for now

    def to_string(self):
        if self.matrix.data:
            status = ("not compressed" if self.matrix.val['m_innerNonZeros'] else "compressed")
        else:
            status = "empty"
        dimensions = "%d x %d" % (self.matrix.rows, self.matrix.cols)
        layout = "row" if self.matrix.row_major else "column"

        return "Eigen::SparseMatrix<%s>, %s, %s major, %s" % (
            self.matrix.inner_type, dimensions, layout, status)


class EigenQuaternionPrinter:
    """Print an Eigen Quaternion"""

    def __init__(self, val):
        # The gdb extension does not support value template arguments - need to extract them by hand
        val_type = val.type
        if val_type.code == gdb.TYPE_CODE_REF:
            val_type = val_type.target()
        self.type = val_type.unqualified().strip_typedefs()
        self.innerType = self.type.template_argument(0)
        self.val = val

        # Quaternions have a struct as their storage, so we need to walk through this
        self.data = self.val['m_coeffs']['m_storage']['m_data']['array']
        self.data = self.data.cast(self.innerType.pointer())

    def cell_data(self, oe):
        offset, element = oe
        return f'[{element}]', (self.data + offset).dereference()

    def children(self):
        return iter(map(self.cell_data, enumerate(('x', 'y', 'z', 'w'))))

    def to_string(self):
        return "Eigen::Quaternion<%s> (data ptr: %s)" % (self.innerType, self.data)


def build_eigen_dictionary():
    pretty_printers_dict[re.compile('^Eigen::Quaternion<.*>$')] = lambda val: EigenQuaternionPrinter(val)
    pretty_printers_dict[re.compile('^Eigen::Matrix<.*>$')] = lambda val: EigenMatrixPrinter("Matrix", val)
    pretty_printers_dict[re.compile('^Eigen::SparseMatrix<.*>$')] = lambda val: EigenSparseMatrixPrinter("Sparse", val)
    pretty_printers_dict[re.compile('^Eigen::Array<.*>$')] = lambda val: EigenMatrixPrinter("Array", val)

    pretty_printers_dict[re.compile('^EigenPrinters::RowFirst$')] = lambda val: MatrixStringifier("RowFirst", val)
    pretty_printers_dict[re.compile('^EigenPrinters::ColFirst$')] = lambda val: MatrixStringifier("ColFirst", val)
    pretty_printers_dict[re.compile('^EigenPrinters::ByRow$')] = lambda val: MatrixPartStringifier("ByRow", val)
    pretty_printers_dict[re.compile('^EigenPrinters::ByCol$')] = lambda val: MatrixPartStringifier("ByCol", val)


def register_eigen_printers(obj):
    """Register eigen pretty-printers with object file obj"""
    if obj is None:
        obj = gdb
    obj.pretty_printers.append(lookup_function)


def lookup_function(val):
    """Look-up and return a pretty-printer that can print the value"""
    val_type = val.type
    # print(val_type, int(val.address) if val.address is not None else 'xxx')

    if val_type.code == gdb.TYPE_CODE_REF:
        val_type = val_type.target()
    val_type = val_type.unqualified().strip_typedefs()

    typename = val_type.tag
    if typename is None:
        return None

    for regex in pretty_printers_dict:
        if regex.search(typename):
            return pretty_printers_dict[regex](val)

    return None


pretty_printers_dict = {}

if not len(pretty_printers_dict):
    build_eigen_dictionary()
