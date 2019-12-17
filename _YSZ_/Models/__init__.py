"""
Models包
常用模型
"""

import _YSZ_._Base_ as base
# --------------------- 子包目录 ---------------------
import _YSZ_.Models.LeNet5
import _YSZ_.Models.ConvNet


# --------------------- 公用方法 ---------------------
def load_state_dict(model, param_path):
    """
    尝试加载模型参数。

    如果模型的参数文件存在，继续训练
    :param model: 模型
    :param param_path: 参数文件路径
    :return: 加载参数之后的模型
    """
    import os
    import torch
    if os.path.exists(param_path):  # 加载模型的参数
        model.load_state_dict(torch.load(param_path))
    return model
