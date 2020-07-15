import time
import numpy as np
#from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent import DQNAgent
from wrappers import wrapper
from utils import get_args

# Take argument
arg = get_args()

# Build env (first level, right only)
env = gym_super_mario_bros.make(arg.env)
env = JoypadSpace(env, RIGHT_ONLY)
env = wrapper(env)
# Parameters
states = (84, 84, 4)
actions = env.action_space.n

# Pham xuan
# Agent
agent = DQNAgent(states=states, actions=actions, max_memory=100000, double_q=True)

# Episodes
eisodes = 101
rewards = []

# Timing
start = time.time()
step = 0

# model path 
model_path = 'models'

# Main loop
#env.reset()
agent.replay(env,model_path,arg.n_replay,plot = True)
#env.reset()

print("Done")
