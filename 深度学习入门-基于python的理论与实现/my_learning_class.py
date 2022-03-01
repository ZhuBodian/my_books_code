import numpy as np
import my_net_class as mnc
import pandas as pd
from sklearn.datasets import fetch_openml
import pickle
import os


# 0 读取数据集
class Datasets:

    # 读取pkl文件
    def __load_pkl(self):
        with open('x.pkl', 'rb') as f:
            x = pickle.load(f)
        with open('t.pkl', 'rb') as f:
            t = pickle.load(f)

        return x, t

    # 保存pkl文件
    def __save_pkl(self, x, t):
        with open('x.pkl', 'wb') as f:
            pickle.dump(x, f)
        with open('t.pkl', 'wb') as f:
            pickle.dump(t, f)

    # mnist数据集
    def mnist(self):
        if os.path.exists('x.pkl'):
            x, t = self.__load_pkl()
        else:
            mnist = fetch_openml('mnist_784', version=1)
            x, t = mnist["data"].values / 255, pd.get_dummies(mnist["target"].array.codes)  # x正则化处理,y独热编码
            t = t.values
            self.__save_pkl(x, t)
        return x, t


# 1 激活函数
class ActivationFunction:

    # sigmoid函数
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # relu函数
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    # softmax函数
    @staticmethod
    def softmax(x):
        # 用z=x-max(x)，代替原始softmax输入，防止softmax函数数值溢出
        # 由于softmax函数特效，对exp指数的增减常量系数C并不会改变函数输出，因此将输入取值区间变换到(-∞,0]，即可有效防止上溢。又因为exp函数大于0，不会出现分母为0，自然消除了下溢。（参考《深度学习》第四章——数值计算）
        x = x - np.max(x, axis=-1, keepdims=True)  # 溢出措施
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# 2 损失函数
class LossFunction:

    # 交叉熵误差
    @staticmethod
    def cross_entropy_error(y, t):
        # y是神经网络输出，t是正确解标签
        # delta = 1e-7
        # return -np.sum(t * np.log(y + delta))
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 如果教师数据是 one-hot-vector，则将其转换为正确标签的索引
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 3 学习算法
class LearningAlgorithms:

    # 梯度计算
    @staticmethod
    def numerical_gradient(f, x):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        # Efficient multi-dimensional iterator object to iterate over arrays.
        # flags=['multi_index']表示对a进行多重索引；op_flags=['readwrite']表示不仅可以对a进行read（读取），还可以write（写入）
        # it.multi_index代表迭代的二元组索引，迭代更新为(0,0),(0,1),(0,2),...,(0,49),...,(783,49)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val  # 恢复值
            it.iternext()  # 表示进入下一次迭代，如果不加这一句的话，输出的结果就一直都是(0,0)

        return grad

    # 梯度下降法
    @staticmethod
    def gradient_descent(f, init_x, lr=0.01, step_num=100):
        # f为要最优化的函数，init_x是初始值，lr是学习率，step_num是梯度法的重复次数
        x = init_x
        for i in range(step_num):
            grad = LearningAlgorithms.numerical_gradient(f, x)
            x -= lr * grad

        return x


# 4 神经层
class NeuralLayers:
    # Affine
    class Affine:
        def __init__(self, W, b):
            self.W = W
            self.b = b
            self.x = None
            self.dw = None
            self.db = None

        def forward(self, x):
            self.x = x

            return np.dot(x, self.W) + self.b

        def backward(self, dout):
            dx = np.dot(dout, self.W.T)
            self.dW = np.dot(self.x.T, dout)
            self.db = np.sum(dout, axis=0)

            return dx

    # ReLU
    class ReLU:
        def __init__(self):
            self.mask = None

        def forward(self, x):
            self.mask = (x <= 0)  # 记录符合条件的索引
            out = x.copy()
            out[self.mask] = 0

            return out

        def backward(self, dout):
            dout[self.mask] = 0
            dx = dout

            return dx

    # SoftmaxWithLoss层
    class SoftmaxWithLoss:

        def __init__(self):
            self.loss = None
            self.y = None  # softmax的输出
            self.t = None  # 监督数据（独热编码）

        def forward(self, x, t):
            self.t = t
            self.y = ActivationFunction.softmax(x)
            self.loss = LossFunction.cross_entropy_error(self.y, self.t)

            return self.loss

        def backward(self, dout=1):
            batch_size = self.t.shape[0]
            dx = (self.y - self.t) / batch_size

            return dx


# 5 早停
class EarlyStopping:
    def __init__(self, input_size, hidden_size, output_size, patience=None):
        self.patience = patience
        # 如果采用早停法，则构造一个变量来储存计算所得的模型参数，返回模型最佳表现时的参数
        self.params_array = {'W1': np.zeros((patience + 1, input_size, hidden_size)),
                             'b1': np.zeros((patience + 1, hidden_size)),
                             'W2': np.zeros((patience + 1, hidden_size, output_size)),
                             'b2': np.zeros((patience + 1, output_size))}
        # 记录计算梯度的次数，左闭，方便多维数组后续赋值
        self.grad_cal_nums = -1
        # 记录最佳参数
        self.best_pars = self.params_array.copy()

    def early_stopping(self, judge_list):
        # 认为指标是越大越好（准确率）
        assert len(judge_list) > self.patience
        idx = self.patience + 1
        temp = judge_list[-idx:]

        if all(temp[1:] < temp[0]):
            for key in 'W1', 'b1', 'W2', 'b2':
                idx = (self.grad_cal_nums + 1) % (self.patience + 1)  # 最优值在环中的坐标
                self.best_pars[key] = self.params_array[key][idx].copy()
            return True

        return False

    def record_par(self, params):
        # 取余操作构造一个环，但是这个环仅是为了节保存空间而构造，具体环上哪个位置的参数最优，需要查看self.__best_par_index
        self.grad_cal_nums += 1
        idx = self.grad_cal_nums % (self.patience + 1)
        for key in 'W1', 'b1', 'W2', 'b2':
            self.params_array[key][idx] = params[key]


# 6 优化器

class Optimizers:
    # SGD
    class SGD:
        def __init__(self, lr=0.01):
            self.lr = lr

        def update(self, params, grads):
            for key in params.keys():
                params[key] -= self.lr * grads[key]

    # momentum
    class Momentum:
        def __init__(self, lr=0.01, momentum=0.9):
            self.lr = lr
            self.momentum = momentum
            self.v = None

        def update(self, params, grads):
            if self.v is None:  # 这样不用初始化传入额外参数了，直接在这里变相“初始化”
                self.v = {}
                for key, val in params.items():
                    self.v[key] = np.zeros_like(val)

            for key in params.keys():
                self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
                params[key] += self.v[key]

    """AdaGrad"""

    class AdaGrad:
        def __init__(self, lr=0.01):
            self.lr = lr
            self.h = None

        def update(self, params, grads):
            if self.h is None:
                self.h = {}
                for key, val in params.items():
                    self.h[key] = np.zeros_like(val)

            for key in params.keys():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
