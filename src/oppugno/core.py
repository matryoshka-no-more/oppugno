#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import ast
import inspect
import astpretty
from .type import *


class Func:
    def __init__(self):
        self.name = ""
        self.raw = ""


class GPUFunc(Func):
    def __init__(self):
        super().__init__()
        self.dims = []
        self.inputs = []
        self.outputs = []
        self.params = []


class PythonFunc(Func):
    def __init__(self):
        super().__init__()
        self.ast = None
        self.params = []


class CudaFunc(GPUFunc):
    def __init__(self):
        super().__init__()


class KernelFunc(GPUFunc):
    def __init__(self):
        super().__init__()


class Cuda:
    python_func = PythonFunc()
    cuda_func = CudaFunc()
    kernel_func = KernelFunc()

    def __init__(self):
        self.dims = []
        self.params = []
        self.inputs = []
        self.outputs = []

    def get_dim_min(self):
        return min([o.dim for o in self.outputs])


class KernelLoop:
    index_start = 0
    index_stop = 0


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
        cuda = Cuda()
        func_python = cuda.python_func
        func_kernel = cuda.kernel_func
        func_cuda = cuda.cuda_func

        code_str = inspect.getsource(method)
        print(code_str)
        code_ast = ast.parse(code_str)
        func_def = code_ast.body[0]
        func_name = func_def.name
        func_args = func_def.args.args
        func_body = func_def.body

        func_python.ast = func_def
        func_python.name = func_name
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

    return func