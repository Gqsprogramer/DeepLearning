import torch.cuda
from torch import nn  # 导入神经网络模块
from torch.utils.data import DataLoader  # 数据包管理工具
from torchvision import datasets  # 数据处理工具，专门用于图像处理的包
from torchvision.transforms import ToTensor  # 数据转换，张数
from matplotlib import pyplot as plt


"""
    下载数据
"""
# datasets.MNIST来加载MNIST数据集作为训练数据集。
# root='data'：指定数据集存储的根目录，可以根据需要进行更改。
# train=True：表示加载训练数据集
# download=True：如果数据集在指定路径中不存在，将自动从官方源下载并保存。
# transform=ToTensor()：指定数据转换操作，将图像数据转换为PyTorch中的Tensor张量格式。
training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),  # 张量
)  # 对于pythorch库能够识别的数据一般是tensor张量

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

figure=plt.figure()