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


def render(template_file, cuda, target=None):
    template = template_env.get_template(template_file)
    result = template.render(cuda=cuda)
    print(result)
