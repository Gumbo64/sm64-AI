import numpy as np
import torch

def precompute_indices_and_labels(n, k=3):
    # Generate the meshgrid of indices
    ix = np.arange(n)
    idx1, idx2 = np.meshgrid(ix, ix, indexing='ij')
    
    # Select only the upper triangle of indices without the diagonal
    mask = idx2 > idx1
    idx1 = idx1[mask]
    idx2 = idx2[mask]
    
    # Generate the labels. If it is within k steps, label it as 1
    labels = idx2 - idx1 <= k
    labels = labels.astype(int)
    
    return idx1, idx2, labels

def generate_pairs_with_labels(tensor):
    # Gather the pairs using the filtered indices
    pairs = tensor[idx1], tensor[idx2]
    
    # Convert the results back to PyTorch tensors
    pairs_tensor = torch.stack(pairs, dim=1)    
    return pairs_tensor

# Example usage:
input_tensor = torch.tensor([5, 3, 12, 7, 43, 51, 6453, 1235, 123541])
n = input_tensor.size(0)
print(input_tensor.dim())
idx1, idx2, labels = precompute_indices_and_labels(n)

pairs_tensor = generate_pairs_with_labels(input_tensor)
print(pairs_tensor)
print(labels)
