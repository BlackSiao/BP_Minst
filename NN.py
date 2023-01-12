import time

import numpy as np
import tensorflow as tf
import torch
from tensorflow.python.ops.confusion_matrix import confusion_matrix

#以下为激活函数
def label_binarizer(label):
    relabel = np.zeros([len(label), 10], dtype=np.int32)
    for i in range(len(label)):
        relabel[i][label[i]] = 1
    return relabel


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):

        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        #程序开始执行，首先初始化weights文件，其初值值[-0.25,0.25]之间
        self.weights = []
        for i in range(1, len(layers) - 1):
            # 如len(layers)layer是一个list[2,2,1]，则len(layer)=3
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)
    #训练函数，Learning_rate为学习率，此处设置为0.15，epochs设置为10000
    def fit(self, X, y, learning_rate=0.15, epochs=10000):
        X = np.atleast_2d(X) #X：数据集,确认是二维，每行是一个实例，每个实例有一些特征值

        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X
        X = temp
        #将X增加一个维度
        y = np.array(y)
        #设置最小损失率和最大正确率的初值
        min_loss=0.8
        max_accury=0.2
        for k in range(epochs):
            start_time = time.monotonic() #开始计时
            print(k + 1)
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            '''
            #正向传播计算各单元的值
            #np.dot代表两参数的内积，x.dot(y) 等价于 np.dot(x,y)
            #即a与weights内积加上偏置求和，之后放入非线性转化function求下一层
            #a输入层，append不断增长，完成所有正向计算
            '''
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]  #y[i]是标签值，a[-1]是正向传播的预测值
            accury=1 - self.get_variance(error)
            loss = self.get_mse(y[i],a[-1])     #根据预测值和标签值求出损失函数
            if(loss<min_loss):
                min_loss=loss
            if(accury>max_accury):
                max_accury=accury
            # 计算输出层的误差，根据最后一层当前神经元的值，反向更新
            deltas = [error * self.activation_deriv(a[-1])] # 输出层的误差
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
            end_time = time.monotonic() #结束计时
            epoch_min, epoch_sec = self.epoch_time(start_time, end_time)
            print(f'Epoch: {k + 1:02} | Epoch Time: {epoch_min}m {epoch_sec}s')  # epochs的训练时间
            print("训练集的损失率：", loss)
            print("训练准确性为： ", 1 - self.get_variance(error))
        print("最优损失率为：", min_loss)
        print("最优正确率: ",max_accury)

            # 使用验证集进行验证
    def verify(self, X, y, learning_rate=0.15, epochs=10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)
        for k in range(epochs):
            print(k+1)
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]  # y[i]是标签值，a[-1]是正向传播的预测值
            loss = self.get_mse(y[i], a[-1])  # 根据预测值和标签值求出损失函数
            # 计算输出层的误差，根据最后一层当前神经元的值，反向更新
            deltas = [error * self.activation_deriv(a[-1])]  # 输出层的误差
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
            print("验证集的损失率：", loss)
            print("验证集准确性为： ", 1 - self.get_variance(error))

    #预测函数
    def predict(self, X):
        self.weights = np.load("weights.npy", allow_pickle=True) #读取训练函数求得的weights.npy文件
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X
        a = temp
        #调用预测函数
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    #定义一个函数来告诉我们一个epoch 需要多长时间
    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins,elapsed_secs

    #定义一个函数来求loss和accury值 loss-采用均分误差计算
    def get_mse(self, records_real, records_predict):
        """
        均方误差 估计值与真值 偏差
        """
        if len(records_real) == len(records_predict):
            return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
        else:
            return None

    def calculate_accuracy(self, y_pred, y):
        m = np.shape(y)[0]
        err = 0.0
        for i in range(m):
            if y[i, 0] != y_pred[i, 0]:
                err += 1
        rate = err / m
        return rate

    def get_average(self, records):
        """
        平均值
        """
        return sum(records) / len(records)

    def get_variance(self, records):
        """
        方差 反映一个数据集的离散程度
        """
        average = self.get_average(records)
        return sum([(x - average) ** 2 for x in records]) / len(records)

