import numpy as np
import my_learning_class as mlc
from collections import OrderedDict


# 单隐层神经网络(隐层神经元数量有hidden_size个，输入输出神经元数量由input_size，out_size决定)
class TwoLayerNet(mlc.EarlyStopping):

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, patience=None, lr=0.01,
                 optimizer='SGD'):
        # input_size、hidden_size、out_size分别是输入，隐藏，输出层神经元数量

        # 初始化权重，W1，W2是第一层、第二层权重
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size), 'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size)}

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = mlc.NeuralLayers.Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = mlc.NeuralLayers.ReLU()
        self.layers['Affine2'] = mlc.NeuralLayers.Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = mlc.NeuralLayers.SoftmaxWithLoss()

        # 如果早停，继承构造
        if patience is not None:
            super(TwoLayerNet, self).__init__(patience=patience, input_size=input_size, hidden_size=hidden_size,
                                              output_size=output_size)

        # 选定优化器
        if optimizer == 'SGD':
            self.optimizer = mlc.Optimizers.SGD(lr=lr)
        else:
            if optimizer == 'Momentum':
                self.optimizer = mlc.Optimizers.Momentum(lr=lr)
            else:
                if optimizer == 'AdaGrad':
                    self.optimizer = mlc.Optimizers.AdaGrad(lr=lr)

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        # x：输入数据，t：监督标签
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):
        # 数值微分，非常慢
        loss_w = lambda W: self.loss(x, t)

        grads = {'W1': mlc.LearningAlgorithms.numerical_gradient(loss_w, self.params['W1']),
                 'b1': mlc.LearningAlgorithms.numerical_gradient(loss_w, self.params['b1']),
                 'W2': mlc.LearningAlgorithms.numerical_gradient(loss_w, self.params['W2']),
                 'b2': mlc.LearningAlgorithms.numerical_gradient(loss_w, self.params['b2'])}

        return grads

    def bp(self, x, t):
        # BP算法计算梯度，并更新网络
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {'W1': self.layers['Affine1'].dW,
                 'b1': self.layers['Affine1'].db,
                 'W2': self.layers['Affine2'].dW,
                 'b2': self.layers['Affine2'].db}

        # 更新网络参数
        self.optimizer.update(params=self.params, grads=grads)  # network.params为字典，可变对象，在python中为传引用

        # 记录网络参数，方便后续早停
        self.record_par(self.params)

    def set_par(self, params):
        # 设置模型参数
        self.layers['Affine1'].W = params['W1'].copy()
        self.layers['Affine1'].b = params['b1'].copy()
        self.layers['Affine2'].W = params['W2'].copy()
        self.layers['Affine2'].b = params['b2'].copy()
