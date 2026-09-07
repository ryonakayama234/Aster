import numpy as np

X = np.array([[1.0, 2.0],
              [3.0, 4.0]])
w = np.array([10.0, 1.0])
assert X.ndim == 2
assert w.shape == (X.shape[1],)
print(X @ w)

