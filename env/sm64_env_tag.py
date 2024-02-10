from .sm64_env import SM64_ENV
import math

def smallest_angle_between(angle1, angle2):
    angle = abs(angle1 - angle2) % (2 * math.pi)
    if angle > math.pi:
        angle = (2 * math.pi) - angle
    return angle

class SM64_ENV_TAG(SM64_ENV):
    def __init__(self, FRAME_SKIP=4, MAKE_OTHER_PLAYERS_INVISIBLE=True, PLAYER_COLLISION_TYPE=0, AUTO_RESET=False, N_RENDER_COLUMNS=2, render_mode="forced", COMPASS_ENABLED=True, HIDE_AND_SEEK_MODE=True, IMG_WIDTH=128, IMG_HEIGHT=72, ACTION_BOOK=[]):
        # placeholder
        self.prev_distances = [0 for _ in range(2000)]
        self.prev_angle_differences = [0 for _ in range(2000)]

        super(SM64_ENV_TAG,self).__init__(FRAME_SKIP=FRAME_SKIP, MAKE_OTHER_PLAYERS_INVISIBLE=MAKE_OTHER_PLAYERS_INVISIBLE, PLAYER_COLLISION_TYPE=PLAYER_COLLISION_TYPE, AUTO_RESET=AUTO_RESET, N_RENDER_COLUMNS=N_RENDER_COLUMNS, render_mode=render_mode, HIDE_AND_SEEK_MODE=HIDE_AND_SEEK_MODE,COMPASS_ENABLED=COMPASS_ENABLED, IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT, ACTION_BOOK=ACTION_BOOK)
        self.agents = [f"hider_{k}" if k < self.MAX_PLAYERS//2 else f"seeker_{k-self.MAX_PLAYERS//2}" for k in range(self.MAX_PLAYERS) ]
        self.possible_agents = self.agents

        self.AGENT_NAME_TO_INDEX = {self.agents[k]: k for k in range(self.MAX_PLAYERS) } 
        self.INDEX_TO_AGENT_NAME = {k: self.agents[k] for k in range(self.MAX_PLAYERS) }

    def reset(self,seed=None,options=None):
        self.prev_distances = [0 for _ in range(self.MAX_PLAYERS//2)]
        self.prev_angle_differences = [0 for _ in range(self.MAX_PLAYERS)]
        return super().reset(seed=None,options=None)
        
    def render(self):
        return super().render(mode="tag")

    
    def calc_rewards(self, gameStatePointers):
        assert self.MAX_PLAYERS % 2 == 0
        for i in range(int(self.MAX_PLAYERS//2)):
            hiderIndex = i
            seekerIndex = i + self.MAX_PLAYERS//2

            hiderState = gameStatePointers[hiderIndex].contents
            seekerState = gameStatePointers[seekerIndex].contents
            hiderPos = [hiderState.posX, hiderState.posY, hiderState.posZ]
            seekerPos = [seekerState.posX, seekerState.posY, seekerState.posZ]

            angleBetweenPlayers = math.atan2(seekerPos[2] - hiderPos[2], seekerPos[0] - hiderPos[0]) 

            # calculating angle from velocity because mario 64 angles are too weird
            # might be better to use the velocity's angle for training anyway? idk. zero vector (no velocity) gives nan and punishes the AI so idk though
            # for seekers, its almost always good to be facing the hider for chasing them but hiders might want to turn around and look at the seeker to see if they are being chased so idk
            hiderAngle = math.atan2(hiderState.velZ, hiderState.velX)
            hiderAngleDifference = smallest_angle_between(angleBetweenPlayers, hiderAngle)
            hiderAngleDifference_delta = hiderAngleDifference - self.prev_angle_differences[hiderIndex]

            angleBetweenPlayers = math.atan2(hiderPos[2] - seekerPos[2], hiderPos[0] - seekerPos[0]) 
            seekerAngle = math.atan2(seekerState.velZ, seekerState.velX)
            seekerAngleDifference = smallest_angle_between(angleBetweenPlayers, seekerAngle)
            seekerAngleDifference_delta = seekerAngleDifference - self.prev_angle_differences[seekerIndex]

            # nan is usually caused by the two players being in the same position
            # if you don't check isnan, then nan eventually gets added to the neural net's weights and they all become nan, killing the network
            
            if math.isnan(seekerAngleDifference_delta):
                seekerAngleDifference_delta = 0
            if math.isnan(hiderAngleDifference_delta):
                hiderAngleDifference_delta = 0

            d = math.dist(seekerPos, hiderPos)
            d_delta = d - self.prev_distances[hiderIndex]   

            # max speed is about 50
            self.rewards[hiderIndex] = (d_delta/(self.FRAME_SKIP * 50)) * 0.8 + (hiderAngleDifference_delta/math.pi) * 0.2
            self.rewards[seekerIndex] = - (d_delta/(self.FRAME_SKIP * 50)) * 0.5 - (seekerAngleDifference_delta/math.pi) * 0.5
            self.prev_distances[hiderIndex] = d
            self.prev_angle_differences[hiderIndex] = hiderAngleDifference
            self.prev_angle_differences[seekerIndex] = seekerAngleDifference
