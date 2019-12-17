"""
Transformer包
数据转换相关的方法
"""

import _YSZ_._Base_ as base
# --------------------- 公用方法 ---------------------
from _YSZ_.Transformer import *

import torch

def enable_cuda(obj):
    '''将变量转换为cuda类型'''
    return base.enable_cuda(obj)


def to_var(self, requires_grad=False):
    '''变量格式转换为Variable'''
    return base.to_var(self, requires_grad=False)