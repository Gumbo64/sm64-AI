import numpy as np
import torch

def combine_hider_seeker_actions(hiderActions, seekerActions):
    t = torch.zeros(2 * hiderActions.shape[0], dtype=hiderActions.dtype)
    t[0::2] = hiderActions
    t[1::2] = seekerActions
    return t
def split_hider_seeker_tensor(tensor):
    # perform the inverse of the above function
    a = tensor[::2]
    b = tensor[1::2]
    return a,b

# Example usage
v1 = torch.tensor([1, 2, 3, 4])
v2 = torch.tensor([5, 6, 7, 8])
ab = combine_hider_seeker_actions(v1, v2)
print("ab", ab)
a,b = split_hider_seeker_tensor(ab)
print("a", a)
print("b", b)
