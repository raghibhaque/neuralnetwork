import numpy as np

X = np.array([[1], [2], [3], [4], [5]])

y = np.array([[0], [0], [0], [1], [1]])

np.random.seed(42)

W = np.random.randn(1, 1)  # weight
b = np.zeros((1, 1))       # bias