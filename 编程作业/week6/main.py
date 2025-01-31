# 1.导入相关库,下载数据
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.optim import SGD,Adam
from torchvision import datasets
# import matplotlib.ticker as mtick
import matplotlib.ticker as mticker 

device='cuda' if torch.cuda.is_avaliable() else 'cpu'

data_folder='./data/FMNIST'
fmnist=datasets.FashionMNIST(data_folder,download=True,train=True)
fmnist_val=datasets.FashionMNIST(data_folder,download=True,train=False)

tr_images=fmnist.data
tr_targets=fmnist.targets

val_images=fmnist_val.data
val_targets=fmnist_val.targets

# 2.构建数据集
class FMNISTDataset(Dataset):
  def __init__(self,x,y):
    # 转换维度
    x=x.float()
    x=x.view(-1,28*28)
    self.x,self.y=x,y
  def __getitem__(self,ix):
    # 对应索引获取元素（仅样本维度）
    x,y=self.x[ix],self.y[ix]
    return x.to(device),y.to(device)
  def __len__(self):
    # 返回样本数量
    return len(self.x)
  
# 3.DataLoader封装数据
def get_data():
  train=FMNISTDataset(tr_images,tr_targets)
  validation=FMNISTDataset(val_images,val_targets)
  trn_dl=DataLoader(train,batch_size=32,shuffle=True)
  # 不设置batch，将整个数据集视作一个batch
  val_dl=DataLoader(validation,batch_size=len(validation),shuffle=True)
  return trn_dl,val_dl

# 4.定义模型(损失函数与优化器)
def get_model():
  model=nn.Sequential(
    nn.Linear(28*28,1000),
    nn.ReLu(),
    nn.Linear(1000,10)
  ).to(device)
  loss_fn=nn.CrossEntropyLoss()
  optimizer=Adam(model.parameters(),lr=1e-02)
  return model,loss_fn,optimizer

# 5.基于batch的训练过程
def train_batch(x,y,model,optimizer,loss_fn):
  # 开启训练模式
  model.train()

  # 梯度清零
  optimizer.zero_grad()
  prediction=model(x)
  batch_loss=loss_fn(prediction,y)
  
  # 求导
  batch_loss.backward()

  #batch步进
  optimizer.step()
  return batch_loss.item() #item表示返回标量

@torch.no_grad()
def accuracy(x,y,model):
  # 开启模型验证过程
  model.eval() 

  prediction=model(x)

  max_values,argmaxs=prediction.max(-1)
  is_correct=argmaxs==y

  return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def val_loss(x,y,model,loss_fn):
  model.eval()
  prediction=model(x)

  val_batch_loss=loss_fn(prediction,y)
  return val_batch_loss.item()

# 训练神经网络
trn_dl,val_dl=get_data()
model,loss_fn,optimizer=get_model()

train_losses,train_accuracies=[],[]
val_losses,val_accuracies=[],[]

# 迭代10次
for epoch in range(20):
  # 打印迭代次数
  print(epoch)

  #计算损失与准确率
  train_epoch_losses,train_epoch_accuracies=[],[]
  for ix,batch in enumerate(trn_dl):
    x,y=batch
    batch_loss=train_batch(x,y,model,optimizer,loss_fn)
    train_epoch_losses.append(batch_loss)
  train_epoch_loss=np.array(train_epoch_losses).mean()

  #计算迭代中的准确率
  for ix,batch in enumerate(trn_dl):
    x,y=batch
    is_correct=accuracy(x,y,model)
    train_epoch_accuracies.append(is_correct)
  # 本来就是numpy数组
  train_epoch_accuracy=np.mean(train_epoch_accuracies)

  for ix,batch in enumerate(iter(val_dl)):
    x,y=batch
    val_is_correct=accuracy(x,y,model)
    validation_loss=val_loss(x,y,model,loss_fn)
  val_epoch_accuracy=np.mean(val_is_correct)
  train_losses.append(train_epoch_losses)
  train_accuracies.append(train_epoch_accuracies)
  val_losses.append(validation_loss)
  val_accuracies.append(val_epoch_accuracy)

# 绘制训练损失与准确率随时间变化,迭代器的标签
epoches=np.arange(20)+1

plt.subplot(121)
plt.plot(epoches,train_losses,'bo',label='Training loss')
plt.plot(epoches,val_losses,'r',label='validation loss')
# 间隔为1
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and Validation loss with Adam')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')

plt.subplot(122)
plt.plot(epoches,train_accuracies,'bo',label='Training loss')
plt.plot(epoches,val_accuracies,'r',label='validation loss')
# 间隔为1
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and Validation accuracy with Adam')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.gca().set_yticklabels(['{:0.f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()
plt.grid('off')



plt.show()