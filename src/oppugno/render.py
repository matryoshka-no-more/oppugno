#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
from jinja2 import Environment, FileSystemLoader

PATH_BASE = os.path.dirname(os.path.realpath(__file__))
PATH_TEMPLATE = os.path.join(PATH_BASE, "templates")
sys.path.append(PATH_BASE)
template_loader = FileSystemLoader(searchpath=PATH_TEMPLATE)
template_env = Environment(loader=template_loader)


def render(template, target=None):
    template = template_env.get_template(template)
    cuda = {
        "name":
        "saxpy",
        "dims": [1 << 10, 1 << 10],
        "inputs": [
            {
                "type": "float*",
                "name": "xarray"
            },
            {
                "type": "float*",
                "name": "yarray"
            },
        ],
        "outputs": [
            {
                "type": "float*",
                "name": "resultarray"
            },
        ],
        "params": [
            {
                "type": "float",
                "name": "alpha"
            },
        ]
    }
    result = template.render(cuda=cuda)
    print(result)


if __name__ == "__main__":
    render("cuda_h.j2")