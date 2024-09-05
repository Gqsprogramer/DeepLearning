import numpy as np
import matplotlib.pyplot as plt
#模拟数据集
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
#模拟功能
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

#设置随机种子
np.random.seed(1)

#随机生成花形数据
#X(2,400),Y(1,400)
X,Y=load_planar_dataset()

"""
绘图
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
plt.show()
"""

"""
shape_X=X.shape
shape_Y=Y.shape
#训练集中样本数量
m=shape_Y[1]

#使用sklearn中内置的logtistic回归观察效果
clf=sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T)
"""

#将匿名函数作为模型输入
"""
线性模型，效果不好
plot_decision_boundary(lambda x:clf.predict(x),X,Y)
plt.title("Logistic Regression")  # 图标题
LR_predictions = clf.predict(X.T)  # 预测结果
print("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +"% " + "(正确标记的数据点所占的百分比)")
plt.show()
"""

#搭建神经网络
"""
1.定义神经网络结构（输入单元的数量，隐藏单元的数量等）。
2.初始化模型的参数
3.循环：
    实施前向传播
    计算损失
    实现向后传播
    更新参数（梯度下降）
"""


def layer_sizes(X,Y):
    """
        参数：
         X - 输入数据集,维度为（输入的数量，训练/测试的数量）
         Y - 标签，维度为（输出的数量，训练/测试数量）

        返回：
         n_x - 输入层的数量
         n_h - 隐藏层的数量
         n_y - 输出层的数量
    """
    n_x=X.shape[0]
    n_h=4
    n_y=Y.shape[0]

    return n_x,n_h,n_y

#初始化模型参数
def initialize_parameters(n_x,n_h,n_y):
    """
        参数：
            n_x - 输入层节点的数量
            n_h - 隐藏层节点的数量
            n_y - 输出层节点的数量

        返回：
            parameters - 包含参数的字典：
                W1 - 权重矩阵,维度为（n_h，n_x）
                b1 - 偏向量，维度为（n_h，1）
                W2 - 权重矩阵，维度为（n_y，n_h）
                b2 - 偏向量，维度为（n_y，1）

    """
    np.random.seed(2)
    #第一层
    #初始化尽可能靠近0，学习率更快
    W1=np.random.randn(n_h,n_x)*0.01
    #b的初始化没有要求，不会对称
    b1 = np.zeros(shape=(n_h, 1))


    #第二成
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    #断言判断维度
    assert(W1.shape==(n_h,n_x))
    assert (b1.shape==(n_h,1))
    assert (W2.shape==(n_y,n_h))
    assert (b2.shape==(n_y,1))

    #输出参数
    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }

    return parameters

#前向传播（计算成本）
def forward_propagation(X,parameters):
    """
        参数：
             X - 维度为（n_x，m）的输入数据。
             parameters - 初始化函数（initialize_parameters）的输出

        返回：
             A2 - 使用sigmoid()函数计算的第二次激活后的数值
             cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型变量
    """
    #后续都将写成循环格式

    #获取参数
    W1=parameters['W1']
    b1=parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    #传播过程
    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)

    #使用断言判断维度
    assert (A2.shape==(1,X.shape[1]))
    #为啥不检查A1？

    #存储中间格式
    cache={
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2
    }

    #输出预测结果与中间参数
    return (A2,cache)


#计算损失
def compute_cost(A2,Y):
    """
        计算方程（6）中给出的交叉熵成本，

        参数：
             A2 - 使用sigmoid()函数计算的第二次激活后的数值
             Y - "True"标签向量,维度为（1，数量）
             parameters - 一个包含W1，B1，W2和B2的字典类型的变量

        返回：
             成本 - 交叉熵成本给出方程（13）
    """

    m=Y.shape[1]

    #计算成本
    #np.multiply(内积，同位相乘)
    logprobs=np.multiply(np.log(A2),Y)+np.multiply((1-Y),np.log(1-A2))
    cost=np.sum(logprobs)/m
    #压缩为一个标量
    cost=float(np.squeeze(cost))

    assert(isinstance(cost,float))

    return cost

#计算后向传播
def backward_propagation(parameters,cache,X,Y):
    """
       使用上述说明搭建反向传播函数。

       参数：
        parameters - 包含我们的参数的一个字典类型的变量。
        cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量。
        X - 输入数据，维度为（2，数量）
        Y - “True”标签，维度为（1，数量）

       返回：
        grads - 包含W和b的导数一个字典类型的变量。
    """

    #提取参数
    m=X.shape[1]
    W1=parameters['W1']
    W2=parameters['W2']

    A1=cache['A1']
    A2=cache['A2']

    #计算梯度
    dz2=A2-Y
    dw2=(1/m)*np.dot(dz2,A1.T)
    #axis表示轴向，keepdims表示是否保持维度
    db2=(1/m)*np.sum(dz2,axis=1,keepdims=True)
    da1=np.dot(W2.T,dz2)

    #分层，第一层往下
    dz1=np.multiply(da1,(1-np.power(A1,2)))
    dw1=(1/m)*np.dot(dz1,X.T)
    db1=(1/m)*np.sum(dz1,axis=1,keepdims=True)

    grads={
        "dw1":dw1,
        "dw2":dw2,
        "db1":db1,
        "db2":db2
    }

    return grads

#学习率更新参数
def update_parameters(parameters,grads,learning_rate=1.2):
    """
        使用上面给出的梯度下降更新规则更新参数

        参数：
         parameters - 包含参数的字典类型的变量。
         grads - 包含导数值的字典类型的变量。
         learning_rate - 学习速率

        返回：
         parameters - 包含更新参数的字典类型的变量。
    """

    #获取参数
    W1,b1=parameters['W1'],parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']

    dw1,db1=grads['dw1'],grads['db1']
    dw2, db2 = grads['dw2'], grads['db2']

    W1=W1-learning_rate*dw1
    W2=W2-learning_rate*dw2
    b1=b1-learning_rate*db1
    b2=b2-learning_rate*db2

    parameters={
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2
    }

    return parameters

#整合模型过程(*重要*)
def nn_model(X,Y,num_iterations,print_cost=False):
    """
        参数：
            X - 数据集,维度为（2，示例数）
            Y - 标签，维度为（1，示例数）
            n_h - 隐藏层的数量（不需要）
            num_iterations - 梯度下降循环中的迭代次数
            print_cost - 如果为True，则每1000次迭代打印一次成本数值

        返回：
            parameters - 模型学习的参数，它们可以用来进行预测。
    """
    np.random.seed(3)

    #获取层内参数
    n_x,n_h,n_y=layer_sizes(X,Y)

    #初始化模型参数
    parameters=initialize_parameters(n_x,n_h,n_y)
    W1=parameters['W1']
    b1=parameters['b1']
    W2=parameters['W2']
    b2=parameters['b2']

    #前向传播；计算成本（可选，为了绘图）；后向传播；更新参数
    for i in range(num_iterations):
        A2,cache=forward_propagation(X,parameters)
        cost=compute_cost(A2,Y)
        grads=backward_propagation(parameters,cache,X,Y)
        parameters=update_parameters(parameters,grads,learning_rate=0.5)

        if print_cost and i%1000==0:
            print("第 ", i, " 次循环，成本为：" + str(cost))

    return parameters

#预测，将概率转换为类别
def predict(parameters,X):
    """
        使用学习的参数，为X中的每个示例预测一个类

        参数：
    		parameters - 包含参数的字典类型的变量。
    	    X - 输入数据（n_x，m）

        返回
    		predictions - 我们模型预测的向量（红色：0 /蓝色：1）

    """

    #预测直接前向即可
    A2,cache=forward_propagation(X,parameters)
    #四舍五入
    prediction=np.round(A2)

    return prediction


#main流程
parameters=nn_model(X,Y,num_iterations=10000,print_cost=True)

#绘制边界,X.T用于x1，x2的方向，求取网格
plot_decision_boundary(lambda x:predict(parameters,x.T),X,Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
#计算最后的成本函数
print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

plt.show()

