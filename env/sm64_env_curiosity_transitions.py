# Based on this paper
#  "Improving Playtesting Coverage via Curiosity Driven Reinforcement Learning Agents" 2021
# https://arxiv.org/pdf/2103.13798.pdf

from .sm64_env import SM64_ENV
import matplotlib.patches as patches

from collections import defaultdict



import torch
import time
import matplotlib.pyplot as plt


class SM64_ENV_CURIOSITY(SM64_ENV):
    def __init__(self, 
            FRAME_SKIP=4,
            MAKE_OTHER_PLAYERS_INVISIBLE=False,
            PLAYER_COLLISION_TYPE=0, 
            AUTO_RESET=False,
            N_RENDER_COLUMNS=4, 
            render_mode="forced", 
            HIDE_AND_SEEK_MODE=False, 
            COMPASS_ENABLED=False, 
            IMG_WIDTH=128,
            IMG_HEIGHT=72, 
            ACTION_BOOK=[],
            NODES_MAX=3000,
            NODE_RADIUS=800,
            NODES_MAX_VISITS=40, 
            NODE_MAX_HEIGHT_ABOVE_GROUND=800,
            
             
            # T_ALPHA = 0.90,
            # R_GAMMA = 0.99
            T_ALPHA = 0.90,
            R_GAMMA = 1.0
    ):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        # stores the nodes' coordinates
        self.N = torch.zeros((NODES_MAX, 3),device=self.device)
        # T[x][y] is the number of times we have gone from node x to node y (x and y are the indexes)
        self.T = torch.zeros((NODES_MAX, NODES_MAX),device=self.device)
        # U is the temporary matrix that collects transitions while the agent is playing (like T)
        self.U = torch.zeros((NODES_MAX, NODES_MAX),device=self.device)
        # V is the vector that stores the reward that the agent will receive upon visiting each node
        self.V = torch.ones(NODES_MAX,device=self.device)
        
        self.T_ALPHA = T_ALPHA
        self.R_GAMMA = R_GAMMA

        # 200 is a placeholder for self.MAX_PLAYERS
        self.prev_node = [0 for i in range(256)]

        # self.T[0][0] = 1 # set the first node to have 1 visit

        self.NODES_MAX = NODES_MAX
        self.NODE_RADIUS = NODE_RADIUS
        self.NODES_MAX_VISITS = NODES_MAX_VISITS
        self.NODE_MAX_HEIGHT_ABOVE_GROUND = NODE_MAX_HEIGHT_ABOVE_GROUND
        self.node_index = 0

        super(SM64_ENV_CURIOSITY,self).__init__(FRAME_SKIP=FRAME_SKIP, MAKE_OTHER_PLAYERS_INVISIBLE=MAKE_OTHER_PLAYERS_INVISIBLE, PLAYER_COLLISION_TYPE=PLAYER_COLLISION_TYPE, AUTO_RESET=AUTO_RESET, N_RENDER_COLUMNS=N_RENDER_COLUMNS, render_mode=render_mode, HIDE_AND_SEEK_MODE=HIDE_AND_SEEK_MODE,COMPASS_ENABLED=COMPASS_ENABLED, IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT, ACTION_BOOK=ACTION_BOOK)

    def calc_agent_rewards(self, gameStatePointers):
        if self.node_index <= 0:
            state = gameStatePointers[0].contents
            pos = torch.FloatTensor([state.posX, state.posY, state.posZ]).to("cpu")
            self.N[0] = pos
            self.node_index = 1
        
        for player in range(self.MAX_PLAYERS):
            state = gameStatePointers[player].contents
            pos = torch.FloatTensor([state.posX, state.posY, state.posZ]).to("cpu")

            distances = torch.cdist(self.N[:self.node_index].to("cpu"), pos.unsqueeze(0))
            closest_node_index = torch.argmin(distances).item()
            # print(distances.shape)
            closest_distance = distances[closest_node_index]
            if closest_distance <= self.NODE_RADIUS:
                self.U[self.prev_node[player]][self.node_index] += 1
                self.rewards[player] = self.V[closest_node_index]
                self.prev_node[player] = closest_node_index

            elif state.heightAboveGround < self.NODE_MAX_HEIGHT_ABOVE_GROUND and pos[1] - state.heightAboveGround > -6000:
                # add the new node's position
                self.N[self.node_index] = pos
                self.U[self.prev_node[player]][self.node_index] = 1
                # hotfix the new node so that it has the max reward
                # so that the reward is defined for the rest of the episode
                self.V[self.node_index] = 1

                self.rewards[player] = 1
                self.prev_node[player] = self.node_index

                self.node_index += 1
            else:
                # if you are too high above the ground then you probably just fell off an edge which is almost always bad
                self.rewards[player] = -1

    def calc_V(self):
        # heavily based on page 80 of "grokking deep reinforcement learning" by Miguel Morales

        # P[x][y] is the probability of going to node y given that we are at node x
        P = torch.zeros((self.node_index, self.node_index),device=self.device)
        # R[x][y] is the reward of going from node x to node y (exponentially decayed)
        R = torch.ones((self.node_index, self.node_index),device=self.device)

        # Apply U to T and then reset U to 0
        self.T = self.T_ALPHA * self.T + (1 - self.T_ALPHA) * self.U
        self.T_size = torch.sum(self.T)
        self.U = torch.zeros((self.NODES_MAX, self.NODES_MAX),device=self.device)

        # calculate P
        for x in range(self.node_index):
            sum_x = torch.sum(self.T[x])
            if sum_x == 0:
                P[x][x] = 1
                continue
            for y in range(self.node_index):
                P[x][y] = self.T[x][y] / sum_x

        # calculate R
        # R = torch.exp(-self.T / self.T_size) / 5
        # R = (1 - self.T / self.T_size)
        
        inward_transitions = torch.zeros((self.node_index),device=self.device)
        for x in range(self.node_index):
            for y in range(self.node_index):
                inward_transitions[y] += self.T[x][y]
        # R = torch.exp(-inward_transitions / self.T_size) / 5
        R = (1 - inward_transitions / self.T_size) / 5
        
        
        # Value iteration

        self.V = torch.zeros(self.NODES_MAX,device=self.device)
        error = 9999999999
        
        timeout = 10000
        count = 0
        while error > 1e-6 and count < timeout:
            V_new = torch.zeros(self.NODES_MAX,device=self.device)
            for x in range(self.node_index):
                V_new[x] = R[x]
                for y in range(self.node_index):
                    done = (y == 0)
                    unfinished = (x == y and self.T[x][x] == 1)
                    # V_new[x] += P[x][y] * ( R[x][y] + self.R_GAMMA * self.V[y] * (not done))
                    V_new[x] += P[x][y] * ( R[x] + self.R_GAMMA * self.V[y] * (not done) + self.R_GAMMA * unfinished)
            error = torch.max(torch.abs(V_new - self.V))
            self.V = V_new
            count += 1
            print(f"Error: {error}")
        print(f"Value iteration took {count} iterations")


    def make_infos(self, gameStatePointers):
        super().make_infos(gameStatePointers)
        for i in range(self.MAX_PLAYERS):
            self.infos[i]["node_index"]= self.node_index

    def reset(self, seed=None, options=None):
        # Game logic
        self.calc_V()
        self.prev_node = [0 for i in range(self.MAX_PLAYERS)]


        # Visualisation
        fig, ax = plt.subplots()
        x = [self.N[k][0].to("cpu").item() for k in range(self.node_index)]
        y = [self.N[k][2].to("cpu").item() for k in range(self.node_index)]
        # Create a colormap based on the values of self.V
        cmap = plt.cm.get_cmap('plasma')
        # Normalize the values of self.V to the range [0, 1]
        norm = plt.Normalize(self.V.min().to("cpu"), self.V.max().to("cpu"))
        # norm = plt.Normalize(0, 10)

        # Plot the circles with color based on self.V
        for i in range(self.node_index):
            circle = patches.CirclePolygon((x[i], y[i]), radius=self.NODE_RADIUS, facecolor=cmap(norm(self.V[i].to("cpu"))), alpha=0.5)
            ax.add_patch(circle)

        # Add a colorbar legend on the side
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Value (self.V)')
        
        ax.set_xlim(-8000, 8000)
        ax.set_ylim(-8000, 8000)
        img = plt.imread("map_BOB.png")
        ax.imshow(img, extent=[-8000, 8000, -8000, 8000])
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_aspect('equal', adjustable='box')
        plt.savefig(f"graphs/transitions/!newest.png")
        plt.savefig(f"graphs/transitions/graph_{int(time.time())}.png")
        plt.close(fig)


        return super().reset(seed, options)
    
    def reset_nodes(self):
        self.node_index = 1
    



# 3D plot that I tried in reset(), but its hard to think where each part is

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x = [self.nodes[k][0].item() for k in range(self.node_index)]
        # y = [self.nodes[k][1].item() for k in range(self.node_index)]
        # z = [self.nodes[k][2].item() for k in range(self.node_index)]

        # # swap y and z
        # scatter = ax.scatter(x, z, y, c='blue', alpha=0.2)

        # ax.set_xlim(-8000, 8000)
        # ax.set_ylim(-8000, 8000)
        # ax.set_zlim(-3000, 3000)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Z")
        # ax.set_zlabel("Y")

        # def update(frame):
        #     ax.view_init(elev=30, azim=frame)  # Rotate the plot by changing the azimuth angle and set a higher elevation
        #     return scatter,

        # ani = animation.FuncAnimation(fig, update, frames=range(0, 360, 5), interval=100)
        # ani.save("rotation_animation.gif", writer='pillow')
        # plt.close(fig)
