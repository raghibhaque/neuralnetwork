import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, W, b):
    z = X @ W + b
    a = sigmoid(z)
    return a

def compute_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def backward(X, y, y_pred):
    m = X.shape[0]
    
    dz = (y_pred - y) * y_pred * (1 - y_pred)
    
    dW = (X.T @ dz) / m
    db = np.sum(dz, axis=0, keepdims=True) / m
    
    return dW, db

X = np.array([[1], [2], [3], [4], [5]])

y = np.array([[0], [0], [0], [1], [1]])

np.random.seed(42)

W = np.random.randn(1, 1)  # weight
b = np.zeros((1, 1))       # bias

y_pred = forward(X, W, b)
print(y_pred)

loss = compute_loss(y_pred, y)
print(loss)

dW, db = backward(X, y, y_pred)

lr = 0.1

W = W - lr * dW
b = b - lr * db

y_pred_new = forward(X, W, b)
loss_new = compute_loss(y_pred_new, y)

print("New loss:", loss_new)

for epoch in range(1000):
    y_pred = forward(X, W, b)
    loss = compute_loss(y_pred, y)
    
    dW, db = backward(X, y, y_pred)
    
    W -= lr * dW
    b -= lr * db
    
    if epoch % 100 == 0:
        print(loss)


print("\nFinal predictions:")
print(forward(X, W, b))

print("\nFinal weight:", W)
print("Final bias:", b)