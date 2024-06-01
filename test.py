from collections import deque
import numpy as np
import math
a = np.array([[1, 2, 3], [4, 5,6]])
b = np.array([[6, 7, 8],[ 9, 10,11]])
c = np.concatenate([a, b], axis=0)
d = np.concatenate([a, b], axis=1)
print(c)
print(d)
rotation_angle = - math.atan2(1, 1)
rotation_matrix = np.array([[np.cos(rotation_angle),-np.sin(rotation_angle), 0],
                            [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                            [0                     , 0                     , 1]])
print(np.dot(c, rotation_matrix))

n = np.array([54,3,99,4,0])
print(np.argsort(n))
print(n[np.argsort(n)[0:2]])

print(math.atan2(0,0))


numerical_input = np.zeros(0)
numerical_input = np.concatenate([numerical_input, np.array([1,2,3])])
print(numerical_input)
