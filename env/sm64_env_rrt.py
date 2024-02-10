# Based on this paper
#  "Improving Playtesting Coverage via Curiosity Driven Reinforcement Learning Agents" 2021
# https://arxiv.org/pdf/2103.13798.pdf
# And also the RRT* algorithm

from .sm64_env import SM64_ENV

import numpy as np
import random
import pickle
from collections import defaultdict


import numpy as np
import torch
import torch
import networkx as nx
import matplotlib.pyplot as plt
import math
def smallest_angle_between(angle1, angle2):
    angle = abs(angle1 - angle2) % (2 * math.pi)
    if angle > math.pi:
        angle = (2 * math.pi) - angle
    return angle



class SM64_ENV_RRT(SM64_ENV):
    def __init__(self, FRAME_SKIP=4, MAKE_OTHER_PLAYERS_INVISIBLE=True, PLAYER_COLLISION_TYPE=0, AUTO_RESET=False, N_RENDER_COLUMNS=4, render_mode="forced", HIDE_AND_SEEK_MODE=False, COMPASS_ENABLED=True, IMG_WIDTH=128, IMG_HEIGHT=72, ACTION_BOOK=[], 
                 NODE_RADIUS= 200,NODES_MAX=20000, SAVE_PATH="", LOAD_PATH=""):
        self.SAVE_PATH = SAVE_PATH
        self.nodes = torch.zeros((NODES_MAX, 3), device = "cpu")
        self.node_visits = torch.zeros(NODES_MAX, device = "cpu")
        # reserve node number 0 as the goal node
        self.nodes[0] = torch.FloatTensor([0, 0, 0], device = "cpu")
        # reserve node number 1 as the starting area node
        self.nodes[1] = torch.FloatTensor([0, 0, 0], device = "cpu")
        self.node_index = 2
        self.G = nx.DiGraph()
        self.NODE_RADIUS = NODE_RADIUS

        # path will be a list of indices into the nodes array, eg [0, 1, 2, 3] means the path is from node 0 to node 1 to node 2 to node 3
        self.path = [1,0]
        # stores indicie in the path each player is up to, eg [0, 1, 2, 3] means player 0 is at the first node of the path, player 1 is at the second node, etc
        self.player_progress_indices = [0 for _ in range(2000)]
        # keep track of the previous node each player was at so that we can make edges
        self.player_prev_nodes = [1 for _ in range(2000)]
        
        self.prev_distances = [0 for _ in range(2000)]
        self.prev_angle_differences = [0 for _ in range(2000)]
        
        super(SM64_ENV_RRT,self).__init__(FRAME_SKIP=FRAME_SKIP, MAKE_OTHER_PLAYERS_INVISIBLE=MAKE_OTHER_PLAYERS_INVISIBLE, PLAYER_COLLISION_TYPE=PLAYER_COLLISION_TYPE, AUTO_RESET=AUTO_RESET, N_RENDER_COLUMNS=N_RENDER_COLUMNS, render_mode=render_mode, HIDE_AND_SEEK_MODE=HIDE_AND_SEEK_MODE,COMPASS_ENABLED=COMPASS_ENABLED, IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT, ACTION_BOOK=ACTION_BOOK)
        self.G.clear()
        self.nodes[1] = torch.FloatTensor(self.infos[1]["pos"], device = "cpu")
        self.node_index = 2
        self.node_visits = torch.zeros(NODES_MAX, device = "cpu")
        if LOAD_PATH != "":
            with open(f"{LOAD_PATH}/rrt.pkl", "rb") as f:
                self.G, self.nodes, self.node_index, self.node_visits = pickle.load(f)
            
        self.player_progress_indices = [0 for _ in range(self.MAX_PLAYERS)]
        self.player_prev_nodes = [1 for _ in range(self.MAX_PLAYERS)]
            


    def calc_rewards(self, gameStatePointers):
        # print(gameStatePointers[0].contents.posX, gameStatePointers[0].contents.posY, gameStatePointers[0].contents.posZ)
        for player in range(self.MAX_PLAYERS):
            state = gameStatePointers[player].contents
            pos = torch.FloatTensor([state.posX, state.posY, state.posZ], device= "cpu")
            distances = torch.cdist(self.nodes[:self.node_index], pos.unsqueeze(0))
            closest_node_index = torch.argmin(distances).item()
            closest_distance = distances[closest_node_index].item()

            goal_node = self.nodes[self.path[self.player_progress_indices[player]]]
            distance_to_goal = torch.norm(goal_node - pos)

            distance_delta = distance_to_goal - self.prev_distances[player]
            self.prev_distances[player] = distance_to_goal

            angle_from_goal = math.atan2(goal_node[2] - pos[2], goal_node[0] - pos[0]) 
            player_angle = math.atan2(state.velZ, state.velX)

            goal_angle_difference = smallest_angle_between(player_angle, angle_from_goal)
            if math.isnan(goal_angle_difference):
                goal_angle_difference = 0

            goal_angle_difference_delta = goal_angle_difference - self.prev_angle_differences[player]
            self.prev_angle_differences[player] = goal_angle_difference           

            self.rewards[player] = 1 - (distance_delta/(self.FRAME_SKIP * 50)) * 0.5 - (goal_angle_difference_delta/math.pi) * 0.5

            # if near any node
            if closest_distance <= self.NODE_RADIUS:
                self.node_visits[closest_node_index] += 1

                if closest_node_index != self.player_prev_nodes[player] and closest_node_index != 0:
                    # add an edge from prev node to current node
                    edge_len = self.node_dist(self.player_prev_nodes[player], closest_node_index)
                    self.G.add_edge(self.player_prev_nodes[player], closest_node_index, length=edge_len)
                    self.player_prev_nodes[player] = closest_node_index

                # if near the intended node
                if closest_node_index == self.path[self.player_progress_indices[player]]:
                    # give maximum reward for reaching the next node on the path
                    self.rewards[player] = 1
                    self.player_progress_indices[player] += 1
                    if self.player_progress_indices[player] >= len(self.path):
                        self.player_progress_indices[player] = len(self.path) - 1

            else:
                # add the new node
                self.nodes[self.node_index] = pos
                self.node_visits[self.node_index] += 1
                # add an edge from prev node to the current position (which is now a node)
                edge_len = self.node_dist(self.player_prev_nodes[player], self.node_index)
                self.G.add_edge(self.player_prev_nodes[player], self.node_index, length=edge_len)
                self.player_prev_nodes[player] = self.node_index

                self.node_index += 1

    def node_dist(self, index1, index2):
        return torch.norm(self.nodes[index1] - self.nodes[index2]).item()
    
    def make_infos(self, gameStatePointers):
        super().make_infos(gameStatePointers)
        for i in range(self.MAX_PLAYERS):
            self.infos[i]["n_nodes"]= self.node_index

    def step(self, actions):
        # if the player has reached the end of the path, then give them a reward and reset them to the beginning of the path
        self.set_compass_targets([self.nodes[self.path[self.player_progress_indices[player]]] for player in range(self.MAX_PLAYERS)])

        return super().step(actions)

    def make_path(self):
        padding = self.NODE_RADIUS * 2
        sampled_node_index = random.choices(range(1, self.node_index), weights=1/(1+self.node_visits[1:self.node_index]**2))[0]
        goal_point = self.nodes[sampled_node_index] + torch.FloatTensor([random.uniform(-padding, padding), 0, random.uniform(-padding, padding)]).to("cpu")
        closest_node_index = torch.argmin(torch.cdist(self.nodes[1:self.node_index], torch.FloatTensor(goal_point).unsqueeze(0))).item() + 1
        if closest_node_index != 1:
            # Find the shortest path from node 1 (spawn point) to the closest node to goal_point
            shortest_path = nx.astar_path(self.G, source=1, target=closest_node_index, heuristic=lambda u, v: self.node_dist(u, v))
        else:
            shortest_path = []

        # add the goal point to the end of the path
        self.nodes[0] = goal_point
        self.path = shortest_path + [0]

    def reset(self, seed=None, options=None):
        self.prev_distances = [0 for _ in range(self.MAX_PLAYERS)]
        self.prev_angle_differences = [0 for _ in range(self.MAX_PLAYERS)]
        # stores indicie in the path each player is up to, eg [0, 1, 2, 3] means player 0 is at the first node of the path, player 1 is at the second node, etc
        self.player_progress_indices = [0 for _ in range(self.MAX_PLAYERS)]
        # keep track of the previous node each player was at so that we can make edges
        self.player_prev_nodes = [1 for _ in range(self.MAX_PLAYERS)]
        self.make_path()

        # ======================== RENDERING AND VISUALISATION ONLY
        if self.SAVE_PATH != "":
            with open(f"{self.SAVE_PATH}/rrt.pkl", "wb") as f:
                pickle.dump([self.G, self.nodes, self.node_index, self.node_visits], f)
        else:
            with open(f"rrt.pkl", "wb") as f:
                pickle.dump([self.G, self.nodes, self.node_index, self.node_visits], f)

        positions = {k: [self.nodes[k][0].item(), self.nodes[k][2].item()] for k in range(1,self.node_index)}
        fig, ax = plt.subplots()
        
        nx.draw(self.G, pos=positions, with_labels=False, font_weight='bold', ax=ax,alpha=0.1, node_size=self.NODE_RADIUS)  # Set node_size to a smaller value, e.g., 10        
        ax.set_xlim(-8000, 8000)
        ax.set_ylim(-8000, 8000)
        img = plt.imread("map.png")
        ax.imshow(img, extent=[-8000, 8000, -8000, 8000])
        for i in range(len(self.path) - 1):
            node1 = self.nodes[self.path[i]]
            node2 = self.nodes[self.path[i + 1]]
            ax.plot([node1[0], node2[0]], [node1[2], node2[2]], color='yellow')

        if self.SAVE_PATH != "":
            plt.savefig(f"{self.SAVE_PATH}/graph_visualization.png")
        else:
            plt.savefig(f"graph_visualization.png")
        plt.close(fig)
        #=================================================================
        
        return super().reset(seed, options, random_moves=1)

    def render(self, render_mode="forced"):
        # if render_mode == "forced_withmap":

        return super().render()
