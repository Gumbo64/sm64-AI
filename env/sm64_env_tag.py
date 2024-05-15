from .sm64_env import SM64_ENV
import numpy as np
import math
import random
def smallest_angle_between(angle1, angle2):
    angle = abs(angle1 - angle2) % (2 * math.pi)
    if angle > math.pi:
        angle = (2 * math.pi) - angle
    return angle

class SM64_ENV_TAG(SM64_ENV):
    def __init__(self,**kwargs):
        # placeholder
        self.prev_distances = [0 for _ in range(2000)]
        self.prev_angle_differences = [0 for _ in range(2000)]
        if "HIDE_AND_SEEK_MODE" in kwargs:
            kwargs.pop("HIDE_AND_SEEK_MODE")
        super(SM64_ENV_TAG,self).__init__(HIDE_AND_SEEK_MODE=True,**kwargs)
        self.agents = [f"hider_{k // 2}" if k % 2 == 0 else f"seeker_{k // 2}" for k in range(self.MAX_PLAYERS) ]
        self.possible_agents = self.agents

        self.AGENT_NAME_TO_INDEX = {self.agents[k]: k for k in range(self.MAX_PLAYERS) } 
        self.INDEX_TO_AGENT_NAME = {k: self.agents[k] for k in range(self.MAX_PLAYERS) }

    def reset(self,seed=None,options=None):
        self.prev_distances = [0 for _ in range(self.MAX_PLAYERS//2)]
        self.prev_angle_differences = [0 for _ in range(self.MAX_PLAYERS)]
        obss, infos = super().reset(seed=None,options=None)
        actions = { name : random.randint(0, self.MAX_PLAYERS - 1) for name in self.agents}
        obss, rews, terminations, truncations, infos = self.step(actions)
        return obss, infos
        
    def render(self):
        return super().render(mode="tag")
    
    def step(self,actions):
        obss, rews, terminations, truncations, infos = super().step(actions)

        new_obs = {}
        for key in obss.keys():
            image_input = obss[key]

            pos = infos[key]["pos"]
            vel = infos[key]["vel"]
            partner_pos = infos[key]["partner_pos"]
            partner_vel = infos[key]["partner_vel"]
            relative_pos = [pos[0] - partner_pos[0], pos[1] - partner_pos[1], pos[2] - partner_pos[2]]
            pos_scaler = 16000
            vel_scaler = self.FRAME_SKIP * 50
            numerical_input = np.array([pos[0] / pos_scaler,
                                        pos[1] / pos_scaler, 
                                        pos[2] / pos_scaler, 
                                        vel[0] / vel_scaler, 
                                        vel[1] / vel_scaler, 
                                        vel[2] / vel_scaler, 
                                        relative_pos[0] * 5 / pos_scaler, 
                                        relative_pos[1] * 5 / pos_scaler, 
                                        relative_pos[2] * 5 / pos_scaler, 
                                        partner_vel[0] / vel_scaler, 
                                        partner_vel[1] / vel_scaler, 
                                        partner_vel[2] / vel_scaler])
            

            new_obs[key] = (image_input, numerical_input)


        terminations = {name: 0 for name in self.agents}
        truncations = {name: 0 for name in self.agents}
        return new_obs, rews, terminations, truncations, infos

    
    def calc_agent_rewards(self, gameStatePointers):
        # assert self.MAX_PLAYERS % 2 == 0
        for i in range(int(self.MAX_PLAYERS)):
            is_hider = (i % 2 == 0)
            my_state = gameStatePointers[i].contents
            my_pos = [my_state.posX, my_state.posY, my_state.posZ]
            partner_pos = self.infos[i]["partner_pos"]
            d = math.dist(my_pos, partner_pos)
            # 16000 is the height/width of the level, decent scaler to use
            d_reward = d / (16000)
            if is_hider:
                self.rewards[i] = d_reward
            else:
                self.rewards[i] = 1 - d_reward

            # hiderState = gameStatePointers[hiderIndex].contents
            # seekerState = gameStatePointers[seekerIndex].contents
            # hiderPos = [hiderState.posX, hiderState.posY, hiderState.posZ]
            # seekerPos = [seekerState.posX, seekerState.posY, seekerState.posZ]

            # angleBetweenPlayers = math.atan2(seekerPos[2] - hiderPos[2], seekerPos[0] - hiderPos[0]) 

            # # calculating angle from velocity because mario 64 angles are too weird
            # # might be better to use the velocity's angle for training anyway? idk. zero vector (no velocity) gives nan and punishes the AI so idk though
            # # for seekers, its almost always good to be facing the hider for chasing them but hiders might want to turn around and look at the seeker to see if they are being chased so idk
            # hiderAngle = math.atan2(hiderState.velZ, hiderState.velX)
            # hiderAngleDifference = smallest_angle_between(angleBetweenPlayers, hiderAngle)
            # hiderAngleDifference_delta = hiderAngleDifference - self.prev_angle_differences[hiderIndex]

            # angleBetweenPlayers = math.atan2(hiderPos[2] - seekerPos[2], hiderPos[0] - seekerPos[0]) 
            # seekerAngle = math.atan2(seekerState.velZ, seekerState.velX)
            # seekerAngleDifference = smallest_angle_between(angleBetweenPlayers, seekerAngle)
            # seekerAngleDifference_delta = seekerAngleDifference - self.prev_angle_differences[seekerIndex]

            # # nan is usually caused by the two players being in the same position
            # # if you don't check isnan, then nan eventually gets added to the neural net's weights and they all become nan, killing the network
            
            # if math.isnan(seekerAngleDifference_delta):
            #     seekerAngleDifference_delta = 0
            # if math.isnan(hiderAngleDifference_delta):
            #     hiderAngleDifference_delta = 0
           
            # d_delta = d - self.prev_distances[hiderIndex]   

            # max speed is about 50
            # self.rewards[hiderIndex] = (d_delta/(self.FRAME_SKIP * 50)) * 0.8 + (hiderAngleDifference_delta/math.pi) * 0.2
            # self.rewards[seekerIndex] = - (d_delta/(self.FRAME_SKIP * 50)) * 0.5 - (seekerAngleDifference_delta/math.pi) * 0.5
            # self.prev_distances[hiderIndex] = d
            # self.prev_angle_differences[hiderIndex] = hiderAngleDifference
            # self.prev_angle_differences[seekerIndex] = seekerAngleDifference
        # print(self.rewards)
