import numpy as np

p = np.array([[0.7, 0.3, 0, 0],
              [0.2, 0.6, 0.2, 0],
              [0, 0, 0.7, 0.3],
              [0, 0, 0, 1]])
print(p @ p @ p)

result = np.linalg.matrix_power(p, 3)

print(result)
