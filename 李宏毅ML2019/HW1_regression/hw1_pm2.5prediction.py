import numpy as np
import pandas as pd


def gradient_decent(X, Y, weight, learning_rate, iterations, lambdaw):
    """
    梯度下降，损失函数使用均方误差
    :param X:样本
    :param Y:样本值
    :param weight:权重
    :param learning_rate:学习率
    :param iterations:迭代次数
    :param lambdaw:正则化值
    :return:
    """

    cost_list = []
    for it in range(iterations):
        y_hat = X.dot(weight)  # (5652,)
        loss = y_hat - Y
        cost = np.sum(loss ** 2) / X.shape[0]
        cost_list.append(cost)
        gradient = X.T.dot(loss) / X.shape[0]
        weight -= learning_rate * gradient

        if it % 1000 == 0:
            print("iteration:{},cost:{}".format(it, cost))
    return weight, cost_list


def Adagrad(X, Y, weight, learning_rate, iterations, lambdaw):
    """
    :param X:
    :param Y:
    :param weight:
    :param learning_rate:
    :param iteration:
    :param lambdaw:
    :return:
    """
    cost_list = []
    gradient_sum = np.zeros(X.shape[1])  # (163,) , weight是（163，）
    for it in range(iterations):
        y_hat = np.dot(X, weight)
        loss = y_hat - Y
        cost = np.sum(loss ** 2) / X.shape[0]
        cost_list.append(cost)
        gradient = np.dot(np.transpose(X), loss) / X.shape[0]
        gradient_sum += gradient ** 2
        sigma = np.sqrt(gradient_sum)
        weight -= learning_rate * gradient / sigma

        if it % 1000 == 0:
            print("iteration:{},cost:{}".format(it, cost))
    return weight, cost_list


def data_processing():
    """加载训练集和测试集"""
    train_file = 'ml2019spring-hw1/train.csv'
    test_file = 'ml2019spring-hw1/test.csv'

    """处理训练集，用ndarray存储"""
    train = pd.read_csv(train_file, encoding='big5')
    train.replace('NR', '0', inplace=True)
    train = train.iloc[:, 3:]
    train_data = []
    for i in range(18):
        train_data.append([])
    for index, row in train.iterrows():
        train_data[index % 18] += list(row.values)
    train_data = np.asarray(train_data).astype(float)
    train_X = []
    train_Y = []
    for month in range(12):
        for i in range(471):
            train_X.append([])
            # 每笔数据第10h的pm2.5
            train_Y.append(train_data[9][480 * month + i + 9])
            # 18中污染物
            for f in range(18):
                # 每一笔数据 前9h数据
                for j in range(9):
                    train_X[471 * month + i].append(train_data[f][480 * month + i + j])

    """"处理测试集，用ndnarray存储"""
    test = pd.read_csv(test_file, header=None, encoding='big5')
    test.replace('NR', '0', inplace=True)
    test = test.iloc[:, 2:]
    test_X = []
    for index, row in test.iterrows():
        if index % 18 == 0:
            test_X.append([])
        test_X[index // 18] += list(row.values)

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X).astype(float)
    return train_X, train_Y, test_X


def train_gd(train_X, train_Y, learning_rate, iterations):
    """训练"""
    bias = np.ones((train_X.shape[0], 1))
    train_X = np.concatenate((train_X, bias), axis=1)  # 扩充X,(5652, 163)
    # 初始化权重weight,和学习率learning_rate
    weight_gd = np.zeros(train_X.shape[1])  # (163,)
    weight_gd, cost_list_gd = gradient_decent(train_X, train_Y, weight=weight_gd,
                                              learning_rate=learning_rate,
                                              iterations=iterations,
                                              lambdaw=0.5)
    return weight_gd, cost_list_gd


def train_ada(train_X, train_Y, learning_rate, iterations):
    # Adagrad
    weight_ada = np.zeros(train_X.shape[1])  # (163, )
    weight_ada, cost_list_ada = Adagrad(X=train_X, Y=train_Y, weight=weight_ada,
                                        learning_rate=learning_rate,
                                        iterations=iterations,
                                        lambdaw=0)

    return weight_ada, cost_list_ada


def predict_gd(X, weight):
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((X, bias), axis=1)
    return np.dot(X, weight)


def predict_ada(X, weight):
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((X, bias), axis=1)
    return np.dot(X, weight)


if __name__ == '__main__':
    train_X, train_Y, test_X = data_processing()
    # gd
    weight_gd, cost_list_gd = train_gd(train_X=train_X, train_Y=train_Y,
                                       learning_rate=1e-6, iterations=10000)
    test_Y = predict_gd(X=test_X, weight=weight_gd)

    # ada
    weight_ada, cost_list_ada1 = train_ada(train_X=train_X, train_Y=train_Y,
                                           learning_rate=5, iterations=10000)
    test_Y_ada = predict_ada(X=test_X, weight=weight_gd)
