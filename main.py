import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, W, b):
    z = X @ W + b
    a = sigmoid(z)
    return a

def compute_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

X = np.array([[1], [2], [3], [4], [5]])

y = np.array([[0], [0], [0], [1], [1]])

np.random.seed(42)

W = np.random.randn(1, 1)  # weight
b = np.zeros((1, 1))       # bias

y_pred = forward(X, W, b)
print(y_pred)