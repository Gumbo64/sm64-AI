import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def precompute_indices_and_labels(n, k=3):
    # Generate the meshgrid of indices
    ix = np.arange(n)
    IDX1, IDX2 = np.meshgrid(ix, ix, indexing='ij')
    
    # Select only the upper triangle of indices without the diagonal
    mask = IDX2 > IDX1
    IDX1 = IDX1[mask]
    IDX2 = IDX2[mask]
    
    # Generate the labels. If it is within k steps, label it as 1
    labels = IDX2 - IDX1 <= k
    labels = labels.astype(int)
    
    return IDX1, IDX2, labels

def generate_pairs_with_labels(tensor):
    # Gather the pairs using the filtered indices
    pairs = tensor[IDX1], tensor[IDX2]
    
    # Convert the results back to PyTorch tensors
    pairs_tensor = torch.stack(pairs, dim=1)    
    return pairs_tensor

class K_STEP_MODEL(nn.Module):
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(6 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat(x1, x2)
        return self.model(x)

def n_choose_2(n):
    return n * (n - 1) // 2

def split_walks_to_k_step_pairs(walks, k):
    # walks: a numpy array of shape (num_players, episode_length, 6)
    # there should be num_players * ( episode_length choose 2 )
    num_players, episode_length, state_length = walks.shape 
    data = np.zeros((num_players * n_choose_2(episode_length), 6 * 2))
    labels = np.zeros((num_players * n_choose_2(episode_length), 1))

    idx = 0


class K_STEP_AI():
    def __init__(self, threshold=0.7, k=5):
        self.model = K_STEP_MODEL()
        self.k = k
        self.threshold = threshold

        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = optim.Adam(self.model.parameters(), lr=1e-4)
    
    def train(self, data, labels, epochs=100):
        for epoch in range(epochs):
            self.optimiser.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimiser.step()
    


class RRT_STAR():
    def __init__(self):
        
        pass

    def generate_path(self, collision_tester, start, goal,
                      max_iter=1000, extend_radius=300, goal_radius=100, goal_sample_rate=0.1):
        self.reset()




    def reset(self):
        self.nodes = []
        self.edges = []
        self.path = []
        self.start = None
        self.goal = None
        self.collision_tester = None



if __name__ == "__main__":
    INIT_HP = {
        "MAX_PLAYERS": 2,
        "NUM_DLL": 2,
        "EPISODE_LENGTH": 1000,
    }
    IDX1, IDX2, labels = precompute_indices_and_labels(10)

    ai = K_STEP_AI()
    ai.train()