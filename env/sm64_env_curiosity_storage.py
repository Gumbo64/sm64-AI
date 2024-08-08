import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import defaultdict

import math

class CURIOSITY_STORAGE:
    def __init__(self) -> None:
        self.index=0
        self.chunk_xz_size=20
        self.chunk_y_size=100
        self.bounding_size= 8192
        self.radius=300
        self.edge_radius=1000

        self.max_visits = 400

        self.F_shape = np.array([2*self.bounding_size // self.chunk_xz_size, 2*self.bounding_size // self.chunk_y_size, 2*self.bounding_size // self.chunk_xz_size])
 
        self.paths = {} # dict of lists
        self.path_indexes = {} # dict of integer indexes

        self.sphere_mask = self.create_mask()
        self.F = np.zeros(shape=self.F_shape, dtype=bool)

    def create_ellipsoid_tensor(self,shape_sphere):
        a, b, c = shape_sphere
        x = np.linspace(-1, 1, a)
        y = np.linspace(-1, 1, b)
        z = np.linspace(-1, 1, c)
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        ellipsoid = (xv)**2 + (yv )**2 + (zv )**2 <= 1
        return ellipsoid

    def create_mask(self,):
        shape_sphere = np.array([2*self.radius // self.chunk_xz_size, 2*self.radius // self.chunk_y_size, 2*self.radius // self.chunk_xz_size])
        # print(shape_rad)
        tensor = self.create_ellipsoid_tensor(shape_sphere)
        indices = np.argwhere(tensor)
        indices -= shape_sphere//2
        return indices

    def add_circle(self, centre):
        centre = np.array(centre, dtype=int)
        indices = self.sphere_mask.copy()
        indices += self.pos_to_index(centre)
        # don't go over or under the bounds
        indices = indices[((indices) >= 0).all(axis=1) & ((indices) < np.array(self.F_shape)).all(axis=1)]
        self.F[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

    def add_circles(self, centres):
        for centre in centres:
            self.add_circle(centre)

    def pos_to_index(self,pos):
        x = (pos[0] + self.bounding_size) // self.chunk_xz_size
        y = (pos[1] + self.bounding_size) // self.chunk_y_size
        z = (pos[2] + self.bounding_size) // self.chunk_xz_size
        return (x,y,z)
    def index_to_pos(self,index):
        x = index[0] * self.chunk_xz_size - self.bounding_size
        y = index[1] * self.chunk_y_size - self.bounding_size
        z = index[2] * self.chunk_xz_size - self.bounding_size
        return np.array([x,y,z])
    def multi_pos_to_index(self, positions):
        indices = np.zeros_like(positions, dtype=int)
        indices[:, 0] = (positions[:, 0] + self.bounding_size) // self.chunk_xz_size
        indices[:, 1] = (positions[:, 1] + self.bounding_size) // self.chunk_y_size
        indices[:, 2] = (positions[:, 2] + self.bounding_size) // self.chunk_xz_size
        return indices
    def multi_index_to_pos(self,index):
        x = index[:, 0] * self.chunk_xz_size - self.bounding_size
        y = index[:, 1] * self.chunk_y_size - self.bounding_size
        z = index[:, 2] * self.chunk_xz_size - self.bounding_size
        return np.array([x, y, z]).T

    def is_pos_visited(self,pos):
        return self.F[self.pos_to_index(pos)]==1
    

    
    def sample_visited_pos(self,n_samples=1):
        visited_indices = np.argwhere(self.F)
        n_samples = min(n_samples, visited_indices.shape[0])
        indices = np.random.choice(visited_indices.shape[0], n_samples, replace=False)
        return self.multi_index_to_pos(visited_indices[indices])
    
    def sample_unvisited_pos(self,n_samples=1):
        unvisited_indices = np.argwhere(self.F==0)
        n_samples = min(n_samples, unvisited_indices.shape[0])
        indices = np.random.choice(unvisited_indices.shape[0], n_samples, replace=False)
        return self.multi_index_to_pos(unvisited_indices[indices])
    
    def sample_pos(self,n_samples=1):
        indices = np.random.randint(0, self.F_shape, (n_samples, 3))
        return indices
    
    def get_possible_edges(self, nodes):
        nodes = np.array(nodes)
        dist_matrix = np.linalg.norm(nodes[:, np.newaxis] - nodes, axis=2)
        edge_indices = np.argwhere(dist_matrix <= self.edge_radius)
        edges = [(nodes[i], nodes[j]) for i,j in edge_indices if i != j]
        return edges
    
    def line_of_sight(self, start, end):
        # Bresenham's line algorithm
        start_3d_index = self.pos_to_index(start) 
        end_3d_index = self.pos_to_index(end)
        x0, y0, z0 = start_3d_index
        x1, y1, z1 = end_3d_index
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        if dx > dy and dx > dz:
            steps = dx
        elif dy > dx and dy > dz:
            steps = dy
        else:
            steps = dz
        if steps == 0:
            return True
        x_inc = (x1 - x0) / steps
        y_inc = (y1 - y0) / steps
        z_inc = (z1 - z0) / steps
        x, y, z = x0, y0, z0
        for _ in range(steps):
            x += x_inc
            y += y_inc
            z += z_inc
            if self.F[int(x), int(y), int(z)] == 0:
                return False
        return True


    def line_checked_edges(self,edges):
        checked_edges = []
        for edge in edges:
            if self.line_of_sight(edge[0], edge[1]):
                checked_edges.append(edge)
        return checked_edges

    def plot_F(self,draw_edges=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Get the coordinates where indices is True
        F_indices = np.argwhere(self.F)
        # x, y, z = F_indices[:, 0], F_indices[:, 2], F_indices[:, 1]
        # Plot circles at the True coordinates
        true_x, true_y, true_z = self.multi_index_to_pos(F_indices).T
        # print(true_x, true_y, true_z)

        # if draw_edges:
        #     sampled_nodes = self.sample_visited_pos(300)
        #     # sampled_nodes = self.multi_index_to_pos(F_indices)
        #     edges = self.get_possible_edges(sampled_nodes)
        #     checked_edges = self.line_checked_edges(edges)
        #     for edge in checked_edges:
        #         # remember z and y are swapped for the graph
        #         ax.plot([edge[0][0], edge[1][0]],[edge[0][2],edge[1][2]],[edge[0][1],edge[1][1]], c='b', alpha=0.1)


        # ax.scatter(true_x, true_z, true_y, c='r', marker='o', alpha=0.005)
        ax.scatter(true_x, true_z, true_y, c='r', marker='o', alpha=1)
        # Set the plot limits
        ax.set_xlim(-self.bounding_size, self.bounding_size)
        ax.set_ylim(-self.bounding_size, self.bounding_size)
        ax.set_zlim(-self.bounding_size, self.bounding_size)
        # Set labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        # Show the plot
        plt.show()
    
    def get_rewards(self, infos):
        player_positions = np.array([info["pos"] for info in infos])
        player_pos_indices = self.multi_pos_to_index(player_positions)
        visits = self.F[player_pos_indices[:, 0], player_pos_indices[:, 1], player_pos_indices[:, 2]]
        rewards = 1 - visits / self.max_visits
        rewards = np.clip(rewards, 0, 1)
        return rewards
    def add_player_positions(self, infos):
        positions = np.array([info["pos"] for info in infos])
        self.add_circles(positions)

    def get_numerical_obs(self, infos):
        size_per_player = 3*2 + 1 # pos, vel, visit_reward
        numerical_obs = np.zeros((len(infos), size_per_player))
        for i, info in enumerate(infos):
            pos = np.array(info["pos"])
            vel = np.array(info["vel"])
            visits = self.F[self.pos_to_index(pos)] / self.max_visits
            numerical_obs[i] = np.concatenate([pos, vel, visits])
        return visits




# Example usage:
if __name__ == "__main__":
    p = CURIOSITY_STORAGE()


    # points = np.random.randint(-8000, 8000, size=(20, 3))

    # Generate arc shape within -8000, 8000
    theta = np.linspace(0, np.pi, 100)
    radius = 8000
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)

    # Combine x, y, z coordinates into points array
    points = np.column_stack((x, y, z))

    # Add circles at the random points
    p.add_circles(points)
    # print(p.F)
    p.plot_F(draw_edges=True)




