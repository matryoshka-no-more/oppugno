#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import ast
import inspect
import astpretty
from time import time
from random import random
from test import say_hello_to
from typing import List

PATH_CURR = os.path.dirname(os.path.realpath(__file__))
PATH_BASE = os.path.dirname(PATH_CURR)
sys.path.append(PATH_BASE)

from oppugno.cuda import Cuda
from oppugno.type import CU_ARRAY, CU_SIZE

ALPHA = 2.0
DEFAULT_ARR_SIZE = 1 << 10


def time_func(method):
    def func(*args, **kw):
        start = time()
        result = method(*args, **kw)
        stop = time()
        duration = stop - start
        print("{}: {:.3f}ms".format(method.__name__, duration * 1000))
        return result

    return func


def parse_func(method):
    def func(*args, **kw):
        code_str = inspect.getsource(method)
        print(inspect.getmembers(method.__code__))
        code_ast = ast.parse(code_str)
        # print(code_str)
        astpretty.pprint(code_ast)
        func_def = code_ast.body[0]
        func_args = func_def.args.args
        func_body = func_def.body

        for arg in func_args:
            arg_name = arg.arg
            arg_type = arg.annotation
            if not arg_type:
                print(
                    "Error: the type of argument '{}' if not indicated".format(
                        arg_name))
                return None

            print(arg_name, arg_type)

        result = method(*args, **kw)
        return result

    return func


def get_data(size=DEFAULT_ARR_SIZE):
    data = [0] * size
    for i in range(size):
        data[i] = random()
    return data


@time_func
@parse_func
def test(size: CU_SIZE, data_x: CU_ARRAY[float], data_y: CU_ARRAY[float],
         alpha: int):
    result = [0.0] * size
    for i in range(size):
        result[i] = alpha * data_x[i] + data_y[i]
    return result


def main():
    size = 1 << 2
    alpha = 2.0
    data_x = get_data(size)
    data_y = get_data(size)
    result = test(size, data_x, data_y, alpha)

    print(result)


if __name__ == "__main__":
    main()