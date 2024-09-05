import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases  # 参见资料包，或者在文章底部copy
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward  # 参见资料包
import lr_utils  # 参见资料包，或者在文章底部copy

np.random.seed(1)

#初始化参数
def initialize_parameters(n_x,n_h,n_y):
    """
        此函数是为了初始化两层网络参数而使用的函数。
        参数：
            n_x - 输入层节点数量
            n_h - 隐藏层节点数量
            n_y - 输出层节点数量

        返回：
            parameters - 包含你的参数的python字典：
                W1 - 权重矩阵,维度为（n_h，n_x）
                b1 - 偏向量，维度为（n_h，1）
                W2 - 权重矩阵，维度为（n_y，n_h）
                b2 - 偏向量，维度为（n_y，1）

    """

    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn((n_y,n_h))*0.01
    b2=np.zeros((n_y,1))

    assert (W1.shape==(n_h,n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters={
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2
    }

    return parameters

#初始化深层隐藏层
def initialize_parameters_deep(layer_dims):
    """
        此函数是为了初始化多层网络参数而使用的函数。
        参数：
            layers_dims - 包含我们网络中每个图层的节点数量的列表

        返回：
            parameters - 包含参数“W1”，“b1”，...，“WL”，“bL”的字典：
                         W1 - 权重矩阵，维度为（layers_dims [1]，layers_dims [1-1]）
                         bl - 偏向量，维度为（layers_dims [1]，1）
    """
    np.random.seed(3)
    parameters={}
    L=len(layer_dims)

    for l in range(1,L):
        #权重初始化，就不乘以0.01了
        parameters['W'+str(l)]=np.ramdom.randn(layer_dims[l],layer_dims[l-1])/np.sqrt(layer_dims[l-1])
        parameters['b'+str(l)]=np.zeros((layer_dims[l]),1)

        #确认维度
        assert (parameters['W'+str(l)].shape==(layer_dims[l],layer_dims[l-1]))
        assert (parameters['b'+str(l)].shape==(layer_dims[l],1))

    return parameters

#前向传播
#1.线性部分
def linear_forward(A,W,b):
    """
       实现前向传播的线性部分。

       参数：
           A - 来自上一层（或输入数据）的激活，维度为(上一层的节点数量，示例的数量）
           W - 权重矩阵，numpy数组，维度为（当前图层的节点数量，前一图层的节点数量）
           b - 偏向量，numpy向量，维度为（当前图层节点数量，1）

       返回：
            Z - 激活功能的输入，也称为预激活参数
            cache - 一个包含“A”，“W”和“b”的字典，存储这些变量以有效地计算后向传递
    """
    Z=np.dot(W,A)+b
    assert (Z.shape==(W.shape[0],A.shape[1]))
    cache=(A,W,b)
    return cache

#2.激活部分,调用1完成总体输出
def linear_activation_forward(A_prev,W,b,activation):
    """
        实现LINEAR-> ACTIVATION 这一层的前向传播

        参数：
            A_prev - 来自上一层（或输入层）的激活，维度为(上一层的节点数量，示例数）
            W - 权重矩阵，numpy数组，维度为（当前层的节点数量，前一层的大小）
            b - 偏向量，numpy阵列，维度为（当前层的节点数量，1）
            activation - 选择在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】

        返回：
            A - 激活函数的输出，也称为激活后的值
            cache - 一个包含“linear_cache”和“activation_cache”的字典，我们需要存储它以有效地计算后向传递
    """
    #根据激活函数类型计算激活值
    if activation=='sigmoid':
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=sigmoid(Z)
    elif activation=='relu':
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=relu(Z)

    assert (A.shape==(W.shape[0],A_prev.shape[1]))
    cache=(linear_cache,activation_cache)

    return A,cache

#多层模型前向传播
def L_model_forward(X,parameters):
    """
       实现[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID计算前向传播，也就是多层网络的前向传播，为后面每一层都执行LINEAR和ACTIVATION

       参数：
           X - 数据，numpy数组，维度为（输入节点数量，示例数）
           parameters - initialize_parameters_deep（）的输出

       返回：
           AL - 最后的激活值
           caches - 包含以下内容的缓存列表：
                    linear_relu_forward（）的每个cache（有L-1个，索引为从0到L-2）
                    linear_sigmoid_forward（）的cache（只有一个，索引为L-1）
    """
    #缓存z,w,b,a
    caches=[]

    #将x作为a0
    A=X

    #计算层的数量
    L=len(parameters)//2

    #遍历中间层，完成relu激活
    for l in range(1,L):
        A_prev=A
        A,cache=linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],'relu')
        caches.append(cache)

    #最后一层使用sigmoid激活
    AL,cache=linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
    caches.append(cache)

    assert (AL.shape==(1,X.shape[1]))

    return AL,caches

#计算成本函数
def compute_cost(AL,Y):
    """
       实施等式（4）定义的成本函数。

       参数：
           AL - 与标签预测相对应的概率向量，维度为（1，示例数量）
           Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）

       返回：
           cost - 交叉熵成本
    """
    m=Y.shape[1]
    cost=-(1/m)*np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),1-Y))

    cost=np.squeeze(cost)
    assert (cost.shape==())

    return cost

#后向传播
#1.线性部分
def linear_backward(dz,cache):
    """
        为单层实现反向传播的线性部分（第L层）

        参数：
             dZ - 相对于（当前第l层的）线性输出的成本梯度
             cache - 来自当前层前向传播的值的元组（A_prev，W，b）

        返回：
             dA_prev - 相对于激活（前一层l-1）的成本梯度，与A_prev维度相同
             dW - 相对于W（当前层l）的成本梯度，与W的维度相同
             db - 相对于b（当前层l）的成本梯度，与b维度相同
    """
    A_prev,W,b=cache
    m=A_prev.shape[1]
    dw=(1/m)*np.dot(dz,A_prev.T)
    db=(1/m)*np.sum(dz,axis=1,keepdims=True)
    da_prev=np.dot(W.T,dz)

    assert (da_prev.shape==A_prev.shape)
    assert (dw.shape==W.shape)
    assert (db.shape==b.shape)

    return da_prev,dw,db

#2.拟合激活函数
#正向与负向传播主要的问题是线性与激活函数的组合过程如何实现
def linear_activation_backward(da,cache,activation='relu'):
    """
        实现LINEAR-> ACTIVATION层的后向传播。

        参数：
             dA - 当前层l的激活后的梯度值
             cache - 我们存储的用于有效计算反向传播的值的元组（值为linear_cache，activation_cache）
             activation - 要在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
        返回：
             dA_prev - 相对于激活（前一层l-1）的成本梯度值，与A_prev维度相同
             dW - 相对于W（当前层l）的成本梯度值，与W的维度相同
             db - 相对于b（当前层l）的成本梯度值，与b的维度相同
    """
    linear_cache,activation_cache=cache
    if activation=='relu':
        dz=relu_backward(da,activation_cache)
        da_prev,dw,db=linear_backward(dz,linear_cache)
    elif activation=='sigmoid':
        dz=sigmoid_backward(da,activation_cache)
        da_prev,dw,db=linear_forward(dz,linear_cache)

    return da_prev,dw,db

