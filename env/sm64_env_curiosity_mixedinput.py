# adds mixed inputs to the curiosity environment

from .sm64_env_curiosity import SM64_ENV_CURIOSITY

from collections import defaultdict

import gymnasium
import functools
import torch
import matplotlib.pyplot as plt
import numpy as np
import math

class SM64_ENV_CURIOSITY_MIXED(SM64_ENV_CURIOSITY):
    def __init__(self, N_CLOSEST_PLAYERS=5, N_CLOSEST_NODES=10, N_PREVIOUS_POSITIONS=20,
                  **kwargs):
        self.N_CLOSEST_PLAYERS = N_CLOSEST_PLAYERS
        self.N_CLOSEST_NODES = N_CLOSEST_NODES
        self.N_PREVIOUS_POSITIONS = N_PREVIOUS_POSITIONS

        self.pos_scaler = 8192
        self.vel_scaler = 50

        super(SM64_ENV_CURIOSITY_MIXED,self).__init__(N_CLOSEST_PLAYERS=N_CLOSEST_PLAYERS, N_CLOSEST_NODES=N_CLOSEST_NODES, N_PREVIOUS_POSITIONS=N_PREVIOUS_POSITIONS, **kwargs)

    def init_with_max_players(self):
        super(SM64_ENV_CURIOSITY_MIXED,self).init_with_max_players()
        self.prev_positions = np.zeros((self.MAX_PLAYERS, self.N_PREVIOUS_POSITIONS, 3)) - self.pos_scaler

    def reset(self,seed=None,options=None):
        obss, infos = super(SM64_ENV_CURIOSITY_MIXED,self).reset(seed=seed,options=options)
        self.prev_positions = np.zeros((self.MAX_PLAYERS, self.prev_positions.shape[1], 3)) - self.pos_scaler
        return obss, infos

    def step(self,actions):
        obss, rews, terminations, truncations, infos = super(SM64_ENV_CURIOSITY_MIXED,self).step(actions)

        new_obs = {}
        for current_agent in obss.keys():
            current_agent_index = self.AGENT_NAME_TO_INDEX[current_agent]
            image_input = obss[current_agent]

            
            pos = np.array(infos[current_agent]["pos"])
            vel = np.array(infos[current_agent]["vel"])


            rotation_angle = - math.atan2(vel[0], vel[2])
            rotation_matrix = np.array([[np.cos(rotation_angle),-np.sin(rotation_angle), 0],
                                        [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                                        [0                     , 0                     , 1]])
            
            # self input
            self_input = np.array([pos[0] / self.pos_scaler,
                                    pos[1] / self.pos_scaler, 
                                    pos[2] / self.pos_scaler,
                                    vel[0] / self.vel_scaler,
                                    vel[1] / self.vel_scaler,
                                    vel[2] / self.vel_scaler,
                                    np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2) / self.vel_scaler])
            
            # Previous n positions
            prev_position_input = self.prev_positions[current_agent_index]

            prev_input_relative = prev_position_input - pos
            prev_input_relative = np.matmul(prev_input_relative, rotation_matrix)

            prev_position_input /= self.pos_scaler
            prev_input_relative = np.tanh(3 * prev_input_relative / self.pos_scaler)



            # Closest n players
            other_players_pos = np.array([infos[name]["pos"] for name in self.agents if name != current_agent])

            highest_index = min(self.MAX_PLAYERS, self.N_CLOSEST_PLAYERS)
            closest_n_players_indices = np.argsort(np.linalg.norm(other_players_pos - pos))[:highest_index]
            closest_n_players_pos = other_players_pos[closest_n_players_indices]

                # filler for when there are not enough players
            if len(closest_n_players_pos) < self.N_CLOSEST_PLAYERS:
                filler_data = np.zeros((self.N_CLOSEST_PLAYERS - len(closest_n_players_pos), 3)) - self.pos_scaler
                closest_n_players_pos = np.concatenate([closest_n_players_pos, filler_data])


            closest_n_players_pos_relative = closest_n_players_pos - pos
            closest_n_players_pos_relative = np.matmul(closest_n_players_pos_relative, rotation_matrix)
            
            closest_n_players_pos /= self.pos_scaler
            closest_n_players_pos_relative = np.tanh(3 * closest_n_players_pos_relative / self.pos_scaler)
            
            # Closest n nodes
            highest_index = min(self.node_index, self.N_CLOSEST_NODES)
            closest_n_nodes_indices = np.argsort(np.linalg.norm(self.nodes[:self.node_index, :3].numpy() - pos, axis=1))[:highest_index]

            closest_n_nodes_pos = self.nodes[:self.node_index, :3].numpy()[closest_n_nodes_indices]
            closest_n_nodes_visits = self.nodes[:self.node_index, 3].numpy()[closest_n_nodes_indices]

                # filler for when there are not enough nodes
            len_closest_n_nodes = len(closest_n_nodes_pos)
            if len_closest_n_nodes < self.N_CLOSEST_NODES:


                filler_data = np.zeros((self.N_CLOSEST_NODES - len_closest_n_nodes, 3)) - self.pos_scaler
                closest_n_nodes_pos = np.concatenate([closest_n_nodes_pos, filler_data])

                filler_data = np.zeros((self.N_CLOSEST_NODES - len_closest_n_nodes))
                closest_n_nodes_visits = np.concatenate([closest_n_nodes_visits, filler_data])



            closest_n_nodes_pos_relative = closest_n_nodes_pos - pos
            closest_n_nodes_pos_relative = np.matmul(closest_n_nodes_pos_relative, rotation_matrix)

            closest_n_nodes_pos /= self.pos_scaler
            closest_n_nodes_pos_relative = np.tanh(3 * closest_n_nodes_pos_relative / self.pos_scaler)
            closest_n_nodes_visits /= self.NODES_MAX_VISITS

            # Concatenate all inputs
            numerical_input = np.concatenate([self_input,
                                              prev_position_input.flatten(), 
                                              prev_input_relative.flatten(), 
                                              closest_n_players_pos.flatten(), 
                                              closest_n_players_pos_relative.flatten(),
                                              closest_n_nodes_pos.flatten(),
                                              closest_n_nodes_pos_relative.flatten(),
                                              closest_n_nodes_visits.flatten()])
            
            # print(self_input.shape, prev_position_input.shape, closest_n_players_pos.shape,  closest_n_nodes_pos.shape, closest_n_nodes_visits.shape)

            new_obs[current_agent] = (image_input, numerical_input.astype(np.float32))

            self.prev_positions[current_agent_index] = np.roll(self.prev_positions[current_agent_index], 1, axis=0)
            self.prev_positions[current_agent_index][0] = pos


        terminations = {name: 0 for name in self.agents}
        truncations = {name: 0 for name in self.agents}

        return new_obs, rews, terminations, truncations, infos
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/

        num = 0
        #self input
        num += 7
        #previous n positions
        num += self.N_PREVIOUS_POSITIONS * 3 * 2
        #closest n players
        num += self.N_CLOSEST_PLAYERS * 3 * 2
        #closest n nodes + visits
        num += self.N_CLOSEST_NODES * 3 * 2 + self.N_CLOSEST_NODES

        numerical_obs = gymnasium.spaces.Box(low=-1, high=1, shape=(num,), dtype=np.float32)
        img_obs = gymnasium.spaces.Box(low=0, high=255, shape=(self.IMG_HEIGHT,self.IMG_WIDTH,3), dtype=np.uint8)
        
        space = gymnasium.spaces.Tuple([img_obs,numerical_obs])
        return space