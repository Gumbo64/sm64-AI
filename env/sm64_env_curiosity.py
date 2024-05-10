# Based on this paper
#  "Improving Playtesting Coverage via Curiosity Driven Reinforcement Learning Agents" 2021
# https://arxiv.org/pdf/2103.13798.pdf

from .sm64_env import SM64_ENV
import matplotlib.patches as patches

from collections import defaultdict


import time
import torch
import matplotlib.pyplot as plt


class SM64_ENV_CURIOSITY(SM64_ENV):
    def __init__(self, NODES_MAX=3000, NODE_RADIUS= 300, 
                 NODES_MAX_VISITS=40, NODE_MAX_HEIGHT_ABOVE_GROUND=800, **kwargs):
        # format of each node is (x,y,z,visits)
        # no need to eat up vram with these nodes, not much faster anyway
        self.nodes = torch.zeros((NODES_MAX, 4), device="cpu")
        self.nodes[0][3] = 1 # set the first node to have 1 visit

        self.NODE_RADIUS = NODE_RADIUS
        self.NODES_MAX_VISITS = NODES_MAX_VISITS
        self.NODE_MAX_HEIGHT_ABOVE_GROUND = NODE_MAX_HEIGHT_ABOVE_GROUND
        self.node_index = 1

        super(SM64_ENV_CURIOSITY,self).__init__(NODES_MAX=3000, NODE_RADIUS= 300, 
                NODES_MAX_VISITS=40, NODE_MAX_HEIGHT_ABOVE_GROUND=800, **kwargs)

    def calc_agent_rewards(self, gameStatePointers):
        # remember the number of visits to each node, then update them all afterwards
        visited_nodes = defaultdict(int)
        # print([ gameStatePointers[i].contents.posY - gameStatePointers[i].contents.heightAboveGround for i in range(self.MAX_PLAYERS)])
        for player in range(self.MAX_PLAYERS):
            state = gameStatePointers[player].contents
            pos = torch.FloatTensor([state.posX, state.posY, state.posZ]).to("cpu")
            distances = torch.cdist(self.nodes[:self.node_index, :3], pos.unsqueeze(0))
            closest_node_index = torch.argmin(distances).item()
            # print(distances.shape)
            closest_distance = distances[closest_node_index]
            if closest_distance <= self.NODE_RADIUS:
                visited_nodes[closest_node_index] += 1

                # choose linear (like the original paper) or exponential
                # self.rewards[player] = 1 - self.nodes[closest_node_index][3].cpu() / self.NODES_MAX_VISITS
                self.rewards[player] = torch.exp(-4 * self.nodes[closest_node_index][3].cpu() / self.NODES_MAX_VISITS)

            elif state.heightAboveGround < self.NODE_MAX_HEIGHT_ABOVE_GROUND and pos[1] - state.heightAboveGround > -6000:
                # print(pos[1] - state.heightAboveGround)
                # add the new node
                self.nodes[self.node_index][:3] = pos
                self.nodes[self.node_index][3] = 0
                visited_nodes[self.node_index] += 1
                self.node_index += 1
                self.rewards[player] = 1
            else:
                # if you are too high above the ground then you probably just fell off an edge which is almost always bad
                self.rewards[player] = -1


        add_array = [visited_nodes[k] for k in range(len(self.nodes))]

        self.nodes[:, 3] += torch.FloatTensor(add_array).to("cpu")

    def make_infos(self, gameStatePointers):
        super().make_infos(gameStatePointers)
        for i in range(self.MAX_PLAYERS):
            self.infos[i]["node_index"]= self.node_index

    def reset(self, seed=None, options=None):
        ##################### GRAPHING
        fig, ax = plt.subplots()
        x = [self.nodes[k][0].item() for k in range(self.node_index)]
        y = [self.nodes[k][2].item() for k in range(self.node_index)]
        visits = [self.nodes[k][3].item() for k in range(self.node_index)]
        cmap = plt.cm.get_cmap('plasma')
        # Normalize the values of self.V to the range [0, 1]
        sorted_visits = sorted(visits)
        norm = plt.Normalize(min(visits), sorted_visits[max(0,len(sorted_visits)-10)])
        # norm = plt.Normalize(0, 10)

        # Plot the circles with color based on self.V
        for i in range(self.node_index):
            circle = patches.CirclePolygon((x[i], y[i]), radius=self.NODE_RADIUS, facecolor=cmap(norm(visits[i])), alpha=0.35)
            ax.add_patch(circle)

        # Add a colorbar legend on the side
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Visits')


        ax.set_xlim(-8000, 8000)
        ax.set_ylim(-8000, 8000)
        img = plt.imread("map_BOB.png")
        ax.imshow(img, extent=[-8000, 8000, -8000, 8000])
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_aspect('equal', adjustable='box')
        plt.savefig(f"graphs/curiosity/!newest.png")
        plt.savefig(f"graphs/curiosity/graph_{int(time.time())}.png")
        plt.close(fig)

        ################################ ACTUAL RESETTING
        self.node_index = 1
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
