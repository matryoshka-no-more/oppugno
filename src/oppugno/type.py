#!/usr/local/bin/python3
# -*- coding: utf-8 -*-


class TypePrimitive:
    INT = "int"
    NULL = "NULL"
    FLOAT = "float"
    DOUBLE = "double"


class TypeBase:
    def __init__(self, name, dim=0, type_base=TypePrimitive.FLOAT):
        self.dim = dim
        self.name = name
        self.type_base = type_base

    def get_full_type(self):
        return self.type_base + "*" * self.dim


class TypeDim(TypeBase):
    def __init__(self, name):
        super().__init__(name, type_base=TypePrimitive.INT)


def new_arg(dim=1):
    arg = TypeBase("", dim)
    return arg


def new_dim():
    arg = TypeDim("")
    return arg


cu_dim = new_dim
cu_param = new_arg
cu_input = new_arg
cu_output = new_arg