# 用于判断my_net_class.TwoLayerNet中的反向传播法是否书写正确（用书写较为简单，但是耗时较长的微分梯度法作为参考）
import numpy as np
import my_learning_class as mlc
import my_net_class as mnc

# 读取数据集
datasets = mlc.Datasets()
x, t = datasets.mnist()
x_train, x_test, t_train, t_test = x[:60000], x[60000:], t[:60000], t[60000:]

network = mnc.TwoLayerNet(input_size=28 * 28, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprob = network.gradient(x_batch, t_batch)

# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprob[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
