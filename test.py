import numpy as np

# Example Nx3 and Mx3 arrays
N_array = np.array([[1, 1, 1], [4, 5, 6], [7, 8, 9]])  # Nx3 array
M_array = np.array([[10, 11, 12], [13, 14, 15]])       # Mx3 array

# Reshape N_array to be Nx1x3 and M_array to be 1xMx3
N_array_reshaped = N_array[:, np.newaxis, :]
M_array_reshaped = M_array[np.newaxis, :, :]

# Add the reshaped arrays, resulting in an NxMx3 array
result = N_array_reshaped + M_array_reshaped

# Reshape the result to be (N*M)x3
result = result.reshape(-1, 3)

print(result)
