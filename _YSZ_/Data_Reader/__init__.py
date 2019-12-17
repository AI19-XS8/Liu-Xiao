"""
Data_Reader:
封装加载数据集的方法
"""

import _YSZ_._Base_ as base
# --------------------- 子包目录 ---------------------
import _YSZ_.Data_Reader.Mnist
import _YSZ_.Data_Reader.Cifar100
import _YSZ_.Data_Reader.Cifar10
import _YSZ_.Data_Reader.SVHN


import torch


# --------------------- 公用方法 ---------------------
def get_dataloader(dataset, batch_size, shuffle=True):
    """
    获取一个数据批量加载器-用于分批次加载数据
    :parameter dataset: 数据集
    :parameter batch_size: 每批加载数据数量
    :parameter shuffle: 是否打乱：True
    """
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle)
    return data_loader
