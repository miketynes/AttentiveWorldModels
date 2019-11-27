import gym
import numpy as np

import matplotlib.pyplot as plt

import time
def show_state(pixels, step):
    """Fn to visualize the agent playing the game in a notebook
    """
    plt.figure(10)
    plt.clf()
    plt.imshow(pixels)
    plt.title("Step: {}".format(step))
    plt.axis('off')
    display.clear_output(wait=True)
    display.display(plt.gcf())
    

    
from IPython import display

env = gym.make("CarRacing-v0")

state = env.reset()
env.close()

ss = []
aa = []
for _ in range(50):
    ss.append(env.render(mode='rgb_array'))
    action = env.action_space.sample()
    a, b, c, d = env.step(action)
    aa.append(action)
env.close()

# for i, (state, action) in enumerate(zip(ss, aa)):
#     show_state(state, i)


a.shape

np.savez_compressed('play_trace.npz', a=aa, b=ss)