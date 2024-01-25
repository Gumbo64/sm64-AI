# Based on this paper
#  "Improving Playtesting Coverage via Curiosity Driven Reinforcement Learning Agents" 2021
# https://arxiv.org/pdf/2103.13798.pdf

from .sm64_env import SM64_ENV

import numpy as np
import random

from collections import defaultdict


import numpy as np
import torch
import torch


class SM64_ENV_CURIOSITY(SM64_ENV):
    def __init__(self, FRAME_SKIP=4, MAKE_OTHER_PLAYERS_INVISIBLE=False, PLAYER_COLLISION_TYPE=0, AUTO_RESET=False, N_RENDER_COLUMNS=4, render_mode="forced", HIDE_AND_SEEK_MODE=False, IMG_WIDTH=128, IMG_HEIGHT=72, ACTION_BOOK=[],
                 MAX_NODES=3000, NODE_RADIUS= 200, MAX_NODE_VISITS=40 ):
        # format of each node is (x,y,z,visits)
        self.nodes = torch.zeros((MAX_NODES, 4),device="cuda" if torch.cuda.is_available() else "cpu")
        self.nodes[0][3] = 1 # set the first node to have 1 visit

        self.node_radius = NODE_RADIUS
        self.node_index = 1
        self.MAX_NODE_VISITS = MAX_NODE_VISITS
        super(SM64_ENV_CURIOSITY,self).__init__(FRAME_SKIP=FRAME_SKIP, MAKE_OTHER_PLAYERS_INVISIBLE=MAKE_OTHER_PLAYERS_INVISIBLE, PLAYER_COLLISION_TYPE=PLAYER_COLLISION_TYPE, AUTO_RESET=AUTO_RESET, N_RENDER_COLUMNS=N_RENDER_COLUMNS, render_mode=render_mode, HIDE_AND_SEEK_MODE=HIDE_AND_SEEK_MODE, IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT, ACTION_BOOK=ACTION_BOOK)

    def calc_rewards(self, gameStatePointers):
        # remember the number of visits to each node, then update them all afterwards
        visited_nodes = defaultdict(int)

        for player in range(self.MAX_PLAYERS):
            state = gameStatePointers[player].contents
            pos = torch.FloatTensor([state.posX, state.posY, state.posZ]).to("cuda" if torch.cuda.is_available() else "cpu")
            distances = torch.cdist(self.nodes[:self.node_index, :3], pos.unsqueeze(0))
            closest_node_index = torch.argmin(distances).item()
            # print(distances.shape)
            closest_distance = distances[closest_node_index]
            if closest_distance <= self.node_radius:
                visited_nodes[closest_node_index] += 1

                # choose linear (like the original paper) or exponential
                # self.rewards[player] = 1 - self.nodes[closest_node_index][3].cpu() / self.MAX_NODE_VISITS
                self.rewards[player] = torch.exp(-4 * self.nodes[closest_node_index][3].cpu() / self.MAX_NODE_VISITS)
            else:
                # add the new node
                self.nodes[self.node_index][:3] = pos
                self.nodes[self.node_index][3] = 0
                visited_nodes[self.node_index] += 1
                self.node_index += 1
                self.rewards[player] = 1


        add_array = [visited_nodes[k] for k in range(len(self.nodes))]

        self.nodes[:, 3] += torch.FloatTensor(add_array).to("cuda" if torch.cuda.is_available() else "cpu")
    def make_infos(self, gameStatePointers):
        self.infos = [{"node_index":self.node_index} for _ in range(self.MAX_PLAYERS)]

    def reset(self, seed=None, options=None):
        self.node_index = 1
        return super().reset(seed, options)
