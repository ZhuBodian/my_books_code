import numpy as np
import my_net_class as mnc
import my_learning_class as mlc
import pandas as pd
from sklearn.datasets import fetch_openml
import pickle
import os

# 读取数据
datasets = mlc.Datasets()
x, t = datasets.mnist()
x_train, x_test, t_train, t_test = x[:60000], x[60000:], t[:60000], t[60000:]  # mnist数据集自动划分好了训练集与验证集

# 超参数
iters_num = 10000  # 最多训练10000次
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
early_stopping_patience = 50  # 能容忍的test_acc不降低的最大轮次

train_loss_list = []
train_acc_list = []
test_acc_list = []

network = mnc.TwoLayerNet(input_size=28 * 28, hidden_size=50, output_size=10, patience=early_stopping_patience,
                          optimizer='AdaGrad', lr=learning_rate)


for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度，并更新参数
    network.bp(x_batch, t_batch)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    train_acc = network.accuracy(x_train, t_train)
    train_acc_list.append(train_acc)

    test_acc = network.accuracy(x_test, t_test)
    test_acc_list.append(test_acc)

    if (i + 1) % 50 == 0:
        print('当前轮次：', i + 1, '；', '损失：', loss, '；', '训练准确率：', train_acc, '；', '测试准确率：', test_acc)

    if i > early_stopping_patience:  # 保证list够长
        if network.early_stopping(judge_list=test_acc_list):
            temp = test_acc_list[-(early_stopping_patience + 1):]
            print(temp)
            break

with open('p160_train_loss_list.pkl', 'wb') as f:
    pickle.dump(train_loss_list, f)

with open('p160_train_acc_list.pkl', 'wb') as f:
    pickle.dump(train_acc_list, f)

with open('p160_test_acc_list.pkl', 'wb') as f:
    pickle.dump(test_acc_list, f)

network.set_par(network.best_pars)
print(network.accuracy(x_test, t_test))
