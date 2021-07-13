import os
import pandas as pd
import numpy as np
import math

class DataManager:
    """
    贝叶斯分类器
    """
    def __init__(self):
        self.data = {}


    def read(self, name, path):
        with open(path, 'r') as f:
            if name == 'Y_train':
                rows = np.array([line.strip('\n') for line in f.readlines()[1:]], dtype=float)
            else:
                rows = np.array([line.strip('\n').split(',') for line in f.readlines()[1:]], dtype=float)
            print(rows.shape)
            if name == 'X_train':
                self.mean = np.mean(rows, axis=0).reshape(1,-1)
                self.std = np.std(rows, axis=0).reshape(1,-1)
                rows = (rows - self.mean) / (self.std + 1e-8)
            elif name == 'X_test':
                rows = (rows - self.mean) / (self.std + 1e-8)
            self.data[name] = rows


    def find_theta(self):
        class_0_id = []
        class_1_id = []
        for i in range(self.data['Y_train'].shape[0]):
            if self.data['Y_train'][i] == 0:
                class_0_id.append(i)
            else:
                class_1_id.append(i)

        class_0 = self.data['X_train'][class_0_id]
        class_1 = self.data['X_train'][class_1_id]

        mean_0 = np.mean(class_0, axis=0)
        mean_1 = np.mean(class_1, axis=0)

        n = class_0.shape[1]
        cov_0 = np.zeros((n,n))
        cov_1 = np.zeros((n, n))
        for i in range(class_0.shape[0]):
            cov_0 += np.dot(np.transpose([class_0[i]-mean_0]), [class_0[i]-mean_0]) / class_0.shape[0]
        for i in range(class_1.shape[0]):
            cov_1 += np.dot(np.transpose([class_1[i]-mean_1]), [class_1[i]-mean_1]) / class_1.shape[0]

        cov = (class_0.shape[0]*cov_0 + class_1.shape[0]*cov_1) / (class_0.shape[0] + class_1.shape[0])
        mean_0_r = np.asarray(mean_0).reshape(-1, 1)
        mean_1_r = np.asarray(mean_1).reshape(-1, 1)
        self.w = np.transpose(mean_0_r - mean_1_r).dot(np.linalg.inv(cov))
        self.b = (-0.5) * (np.transpose(mean_0_r).dot(np.linalg.inv(cov)).dot(mean_0_r)) \
                 + (0.5) * (np.transpose(mean_1_r).dot(np.linalg.inv(cov)).dot(mean_1_r)) \
                 +  np.log(np.float(class_0.shape[0]) / np.float(class_1.shape[0]))



    def predict(self):
        Y_test = []
        for i in range(self.data['X_test'].shape[0]):
            z = self.w.dot(self.data['X_test'][i])  + self.b
            label = np.round(self.sigmoid(z)).astype(np.int)
            Y_test.append(abs(label[0][0] - 1))
        return Y_test


    def sigmoid(self, z):
        """
        激活函数
        :param z: 参数大于0时，函数值大于0.5；
                   参数小于0时，函数值小于0.5。
        :return:
        """
        return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


if __name__ == '__main__':
    dm = DataManager()
    dm.read('X_train', 'dataset/X_train.txt')
    dm.read('X_test', 'dataset/X_test.txt')
    dm.read('Y_train', 'dataset/Y_train.txt')
    dm.find_theta()
    Y_test = dm.predict()
    df = pd.DataFrame(Y_test, columns=['label'])
    df.to_csv('dataset/Y_test_hat.csv', index=False)