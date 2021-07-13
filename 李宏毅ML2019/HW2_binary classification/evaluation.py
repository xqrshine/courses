import pandas as pd
import numpy as np

# 准确度
Y_test_hat = np.array(pd.read_csv('dataset/Y_test_hat.csv')['label'].values)
Y_test = np.array(pd.read_csv('dataset/Y_test.txt')['label'].values)
print(1 - np.mean(abs(Y_test_hat-Y_test)))

# 0.843744241754192
