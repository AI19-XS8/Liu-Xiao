import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class SVHN_dataset():
    """
    SVHN 数据集加载类
    """

    def __init__(self, path='./Data/'):
        self.path = path
        self.download = True
        if os.path.exists(path) and os.listdir(path):  # 判断数据是否存在
            self.download = False

    def get_train_dataset(self):
        '''获取训练数据'''
        train_dataset = datasets.SVHN(root=self.path, split='train', download=self.download,
                                         transform=transforms.ToTensor())  # 训练数据
        return train_dataset

    def get_test_dataset(self):
        '''获取测试数据'''
        test_dataset = datasets.SVHN(root=self.path, split='test', download=self.download,
                                        transform=transforms.ToTensor())  # 测试数据
        return test_dataset

    def get_dataset(self):
        '''获取训练和测试数据'''
        train_dataset = self.get_train_dataset()
        test_dataset = self.get_test_dataset()
        return train_dataset, test_dataset
