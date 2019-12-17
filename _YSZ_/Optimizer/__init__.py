"""
Optimizer包
封装优化相关的方法
"""

import _YSZ_._Base_ as base
# --------------------- 子包目录 ---------------------


# --------------------- 公用方法 ---------------------
def test(model, loader):
    """
    进行一轮测试。

    直接输出模型性能结果
    :param model: 模型
    :param loader: 数据加载器
    :return: 模型性能：正确数量，总样本数，正确率
    """
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)

    for x, y in loader:
        x_var = x.cuda()
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
    acc = float(num_correct) / float(num_samples) * 100
    return num_correct, num_samples, acc


def pred_batch(x, model):
    """
    预测一批数据，返回结果
    :param x: 输入数据
    :param model: 模型
    :return: 预测结果的最大值
    """
    x = x.cuda()
    scores = model(x)
    _, preds = scores.data.cpu().max(1)
    return preds


def wait(seconds):
    """
    程序等待
    :param seconds: 秒
    :return: /
    """
    import time
    time.sleep(seconds)
