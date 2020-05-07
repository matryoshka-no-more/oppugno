#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import ast
import inspect
import astpretty
from .type import *
from .render import render

DEVICE_BLOCK_DIMS = ["x", "y", "z"]
OPPU_REGISTRY = {}


class Cuda:
    def __init__(self, name):
        self.name = name
        self.dims = []
        self.params = []
        self.inputs = []
        self.outputs = []

    def get_dim_min(self):
        return min([o.dim for o in self.outputs])

    def get_args(self):
        return self.dims + self.inputs + self.outputs + self.params

    def get_args_python(self):
        return ", ".join([a.name for a in self.get_args()])

    def get_args_cuda(self):
        return ", ".join([
            "{} {}".format(a.get_full_type(), a.name) for a in self.get_args()
        ])

    def get_cython_func_name(self):
        return "{}_cython".format(self.name)

    def get_cython_func_import(self):
        return "from oppugno_cuda import {}".format(
            self.get_cython_func_name())

    def get_cython_func_call(self):
        return "{}({})".format(self.get_cython_func_name(),
                               self.get_args_python())

    def exec_cython(self):
        cmd = "{}\n{}".format(self.get_cython_func_import,
                              self.get_cython_func_call)
        exec(cmd)


def parse_ast_arg(arg_name, arg_type, cuda):
    if not arg_type:
        return

    if type(arg_type) == ast.Name and arg_type.id.endswith("cu_dim"):
        arg = TypeDim(arg_name)
        cuda.dims.append(arg)
    elif type(arg_type) == ast.Call:
        anno = arg_type.func.id if type(
            arg_type.func) == ast.Name else arg_type.func.attr
        arg = TypeBase(arg_name)

        for k in arg_type.keywords:
            if k.arg == "dim":
                arg.dim = k.value.n
            elif k.arg == "type_base":
                arg.type_base = k.value.s

        if anno.endswith("cu_input"):
            cuda.inputs.append(arg)
        elif anno.endswith("cu_output"):
            cuda.outputs.append(arg)
        elif anno.endswith("cu_param"):
            cuda.params.append(arg)


def cuda(method):
    def func(*args, **kw):
        code_str = inspect.getsource(method)
        print(code_str)
        code_ast = ast.parse(code_str)
        func_def = code_ast.body[0]
        func_name = func_def.name
        func_args = func_def.args.args
        func_body = func_def.body

        cuda = OPPU_REGISTRY.get(func_name)
        if cuda:
            cuda.exec_cython()

        cuda = Cuda(func_name)
        astpretty.pprint(code_ast)

        for func_arg in func_args:
            arg_name = func_arg.arg
            arg_type = func_arg.annotation
            parse_ast_arg(arg_name, arg_type, cuda)

        kernel_depth = 0
        for statement in func_body:
            if type(statement) == ast.AnnAssign:
                arg_name = statement.target.id
                arg_type = statement.annotation
                parse_ast_arg(arg_name, arg_type, cuda)
            elif type(statement) == ast.For:
                if kernel_depth == 0:
                    kernel_depth = cuda.get_dim_min()
                    print("kernel_depth: {}".format(kernel_depth))
                targets = []
                iters = []
                st = statement
                for _ in range(kernel_depth):
                    targets.append(st.target.id)
                    iters.append(st.iter.args[-1].id)
                    st = st.body[0]
                targets.append(st.target.id)
                iters.append(st.iter.args[-1].id)
                # for s in st.body:

        print("inputs: {}".format(len(cuda.inputs)))
        for arg in cuda.inputs:
            print(arg.name, arg.get_full_type())
        print("outputs: {}".format(len(cuda.outputs)))
        for arg in cuda.outputs:
            print(arg.name, arg.get_full_type())
        print("params: {}".format(len(cuda.params)))
        for arg in cuda.params:
            print(arg.name, arg.get_full_type())
        print("dims: {}".format(len(cuda.dims)))
        for arg in cuda.dims:
            print(arg.name, arg.get_full_type())

        render("cuda_h.j2", cuda)

    return func
