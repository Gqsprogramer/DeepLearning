# Pytorch框架

`注：Pytorch安装过程尽量使用pip，不要使用conda`

## 1.导包文件

```python
import torch.cuda
from torch import nn         #导入神经网络模块
from torch.utils.data import DataLoader     #数据包管理工具
from torchvision import datasets         #数据处理工具，专门用于图像处理的包
from torchvision.transforms import ToTensor      #数据转换，张数
```

`pytorch在线文档:`https://pytorch-cn.readthedocs.io/zh/latest/

## 2.基础知识

**基础创建过程**

```python
import torch

x=[[1,2,3],[4,5,6],[7,8,9]]
y=[[5],[6],[7],[8]]

X=torch.tensor(x).float()
Y=torch.tensor(y).float()

print(X)
print(Y)

"""
tensor([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]])
tensor([[5.],
        [6.],
        [7.],
        [8.]])
"""
```

**CPU、GPU运算切换**

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = X.to(device)
Y = Y.to(device)
```

**神经网络框架**

```python
from torch import nn

class MyNeuralNet(nn.Modle):
    def __init__(self):
        super().__init__() #父类nn.modle的初始化过程
        #定义内部网路结构
        self.input_to_hidden_layer=nn.Linear(2,8)
        self.hidden_layer_activation=nn.ReLu()
        self.hidden_to_output_layer=nn.Linear(8,1)
    #前向传播（pytorch保留字段）
    def forward(self,x):
        self.input_to_hidden_layer=nn.Linear(2,8) 
        	#输入2个特征，输出8个特征,包含对应网络的偏执参数b
        self.hidden_layer_activation=nn.ReLu() #ReLu激活函数
        self.hidden_to_output_layer=nn.Linear(8,1) #输入8个特征，输出1个特征
```

等同于

```python
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.parameter(torch.rand(2,8))
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.parameter(torch.rand(8,1))
    def forward(self, x):
        x = x @ self.input_to_hidden_layer
        x = self.hidden_layer_activation(x)
        x = x @ self.hidden_to_output_layer
        return x
```

**创建对象，返回参数**

```python
mynet=MyNeuralNet().to(device) #创建网络对象并传入GPU
```

`打印参数`

```python
print(mynet.input_to_hidden_layer.weight)
"""
初始参数为随机参数，但未进行权重初始化
Parameter containing:
tensor([[ 0.0984,  0.3058],
        [ 0.2913, -0.3629],
        [ 0.0630,  0.6347],
        [-0.5134, -0.2525],
        [ 0.2315,  0.3591],
        [ 0.1506,  0.1106],
        [ 0.2941, -0.0094],
        [-0.0770, -0.4165]], device='cuda:0', requires_grad=True)

"""
```

`获取所有参数`

```python
#生成器对象mynet.parameters() 
for param in mynet.parameters():
    print(param)
```

**定义损失函数**

```python
loss_func=nn.MSELoss()

_Y=mynet(X)
loss_value=loss_func(_Y,Y)
print(loss_value)
```

*踩坑指南：*

*1.输入数据与模型必须在同一个设备上(GPU、CPU),否则会报错<font color='red'> Expected all tensors to be on the same device, but found at least two devices</font>*

*2.输入数据的维度要求X与Y与理论网络相反，详情可见上述代码*

*3.损失计算过程中，传入数据的顺序必须是<font color='red'>（预测数据_Y,真实数据Y）</font>*

**梯度下降**

```python
from torch.optim import SGD
# 梯度下降优化器
opt=SGD(mynet.parameters(),lr=0.001)

# 迭代每个epoch
loss_history=[]
#_表示占位符，不使用range（50）中的值
for _in range(50):
    opt.zero_grad()
    loss_value=loss_func(mynet(X),Y)
    loss_value.backward()
    opt.step()
    loss_history.append(loss_value.cpu().detach().numpy())
```

*踩坑指南：*

*1.在记录对应loss_value时，由于数据在GPU上，所以需要将数据进行依次cpu数据转移，同时解除梯度链接后才能numpy(),再者才能使用plt.plot*

### 总结

基础神经网络流程如下：

1.创建Net实例(创建各层结构，前向传播过程)`class MyNerualNet()...`

2.创建损失函数(nn.MSELoss())

3.创建优化器(torch.optim SGD随机梯度优化)`SGD(myNey.parameters(),lr=0.0001)`

4.迭代优化(梯度清除、计算损失函数、后向传播、步进、增加历史)

## 3.神经网络数据加载(mini-batch)

```python
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn

# 初始化数据
x=[[1,2],[3,4],[5,6],[7,8]]
y=[[3],[7],[11],[15]]

X=torch.tensor(x).float()
Y=torch.tensor(y).float()

#分配数据
device='cuda' if torch.cuda.is_available() else 'cpu'
X=X.to(device)
Y=Y.to(device)
```

**创建数据集**

```python
# 继承父类Dataset
class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x=x.clone().detach() #创建不需要梯度的tensor
        self.y=y.clone().detach() #创建不需要梯度的tensor
    def __len__(self):
        return len(self.x) #返回第一维度长度，即样本数量
    def __getitem__(self,ix):
        return self.x[ix],self.y[ix] #返回ix位置的样本数据
```

**通过DataLoader获取对应batch**

```python
dl=DataLoader(ds,batch_size=2,shuffle=True)
#ds表示通用数据集，batch_size表示一次取多少样本数据，shuffle表示是否在每次提取前打乱数据
# dl对象是可以迭代的
for x,y in dl:
    print(x,y)
```

**定义神经网络**

```python
class myNet(nn):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer=nn.Linear(2,8)
        self.hidden_layer_actiavtion=nn.ReLU()
        self.hidden_to_output_layer=nn.Linear(8,1)
    def forward(self,x):
        x=self.input_to_hidden_layer(x)
        x=self.hidden_layer_actiavtion(x)
        x=self.hidden_to_output_layer(x)
        return x
```

**定义随机梯度、损失函数**

```python
import time
#创建实例、损失函数
mynet=myNet().to(device)
loss_func=nn.MSELoss()

#定义优化器
opt=SGD(mynet.parameters(),lr=0.0001)

loss_history=[]
#开始计时
start=time.time()
for _ in range(50):
    for x,y in dl:
        opt.zero_grad()
        loss_value=loss_func(mynet(x),y)
        loss_value.backward()
        opt.step()
        loss_history.append(loss_value.cpu().detach().numpy())
end=time.time()

print(end-start)
```

### 总结

1.创建自定义数据集，包括`__init__`、`__len__`、`__getitem__`

2.封装DataLoader，包括ds数据集、batch数量、shuffle是否每次乱序

3.需要注意的对象(myNerualNet,loss_func,opt)

## 4.查看中间层输出

**方法1**

```python
print(mynet.hidden_layer_activation(mynet.input_to_hidden_layer(X)))
```

*注意参数输入与层名称的对应顺序*

**方法2**

```python
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer=nn.Linear(2,8)
        self.hidden_layer_activation=nn.ReLu()
        self.hidden_to_output_layer=nn.Linear(8,1)
    def forward(self,x):
        hidden1=self.input_to_hidden_layer(x)
        hidden2=self.hidden_layer_activation(hidden1)
        x=self.hidden_to_output_layer(hidden2)
        return x,hidden1,hidden2
```

## 5.Sequential类搭建神经网络

**原始搭建网络类的过程**

```python
class MyNerualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(2,8)
        self.layer2=nn.ReLu()
        self.layer3=nn.Linear(8,1)
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x
```

**借Sequential类的过程**

```python
model=nn.Sequential(
	nn.Linear(2,8),
    nn.ReLu(),
    nn.Linear(8,1)
).to(device)
```

**打印对应模型摘要(注意安装对应包)**

<font color='red'>需要执行</font>

```python
from torchsummary import summary

print(summary(model,(2,)))
```

**定义损失函数，训练模型**

```python
loss_func=nn.MSEloss()
from torch.optim import SGD
opt=SGD(model.parameters(),lr=0.001)
import time
loss_history=[]
start=time.time()
for _ in range(50):
    for ix,iy in dl:
        opt.zero_grad()
        loss_value=loss_func(model(ix),iy)
        loss_value.backward()
        opt.step()
        loss_history.append(loss_value)
end=time.time()
primt(end-start)
```

**测试集验证**

```python
val=[[9,10],[11,20],[12,30]]

val=torch.tensor(val).float()
print(model(val.to(device)))
```

## 6.模型参数的保存与加载

**显示各层参数**

<font color='red'>需要执行</font>

```python
print(model.state_dict())

"""
执行结果:
"""
```

**模型保存**

```python
save_path='mymodel.pth'

# model.to('cpu') 便于其他机器直接从cpu上获取参数
torch.save(model.state_dict(),save_path)
```

**模型加载**

```python
# 创建空模型
model=nn.Sequential(
	nn.Linear(2,8),
    nn.ReLu(),
    nn.Linear(8,1)
).to(device)

# load文件
state_dict=torch.load('mymodel.pth')

# 加载参数并进行预测
model.load_state_dict(state_dict)
model.to(device)

val=[[9,10],[10,11],[11,12]]
val=torch.tensor(val).float()
print(model(val.to(device)))
```

