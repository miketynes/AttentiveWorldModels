#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import warnings
warnings.filterwarnings('ignore')
import imageio
from skimage.transform import resize


# In[ ]:


import time

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import os
from utils import preprocess_images


# In[ ]:


from models import V as buildVision
from models import V_inverse as Decoder
from models import M as buildMemory
from models import Controller


# In[ ]:


import mdn


# In[ ]:


import matplotlib.pyplot as plt
from IPython import display


# In[ ]:


print('loading models with best weights')


# In[ ]:


V = buildVision()
V_inv = Decoder()


# In[ ]:


V.load_weights('weights/2019.12.07/encoder_weights')
V_inv.load_weights('weights/2019.12.07/decoder_weights')


# In[ ]:


print('model summaries\n')


# In[ ]:


print(V.summary())


# In[ ]:


print(V_inv.summary())


# In[ ]:


M = buildMemory('weights/2019.12.07/mdn_rnn_weights')
get_hidden = K.function(M.layers[0].input, M.layers[0].output)


# In[ ]:


print(M.summary())


# In[ ]:


controller = Controller(32+256, 3)
controller.set_weights(np.load('./weights/C_weights.npy'))


# $$\text{Controller}: \mathbb R^{288} \rightarrow \mathbb R^3 $$

# In[ ]:


print('controller shape:')


# In[ ]:


print(controller.shape)


# In[ ]:


import gym


# In[ ]:


env = gym.make("CarRacing-v0")


# In[ ]:


state = preprocess_images(env.reset())
env.close()


# In[ ]:


def rollout(controller, playback=False):
    if playback:
        ims = []
    state = preprocess_images(env.reset())
    
    M.reset_states()
    h = np.zeros(256)
    done = False
    cumulative_reward = 0
    
    while not done:
        _state = np.zeros((128, 64, 64, 3))
        _state[0] = state
        
        if playback:
            ims.append(state)
        z = V(_state)[2][0] #extract z and first from sequence

        # combine V latent space with M hidden space 
        combined = np.concatenate([z, h], axis=0)
        
        a = controller(combined)
        
        state, reward, done, info = env.step(a)
        state = preprocess_images(state)
        
        cumulative_reward += reward
        
        # extract hidden state from LSTM
        h = get_hidden(tf.expand_dims(tf.expand_dims(np.concatenate([z, a], 0), 0), 0)).squeeze()
        
        # get factored gaussians
        # by feeding current latent_state + action
        z = M(tf.expand_dims(tf.expand_dims(np.concatenate([z, a]), 0), 0))
        
        # sample from factored gaussians
        # 32 = output_dims
        # 5  = num_mixtures
        z = np.apply_along_axis(mdn.sample_from_output, 1, z[0], 32, 5, temp=1.0).squeeze()
    
    env.close()
    if playback:
        return cumulative_reward, ims
    return cumulative_reward


# In[ ]:


print('Rolling out environment. This might take a while...')


# In[ ]:


start = time.time()
r, ims = rollout(controller, playback=True)
end = time.time() - start


# In[ ]:


print('Done.')


# In[ ]:


print('Cumulative reward:')


# In[ ]:


print(r)


# In[ ]:


def show_state(env, step=0, name="", info="", image=None):
    """Fn to visualize the agent playing the game in a notebook
    """
    plt.figure(10)
    plt.clf()
    if image is not None:
        im = image
    else:
        im = env.render(mode="rgb_array")[0]
    plt.imshow(im)
    plt.title("{} | Step: {} {}".format(name, step, info))
    plt.axis('off')
    display.clear_output(wait=True)
    display.display(plt.gcf())


# In[ ]:


#ims = np.load("AICAR.npy")


# In[ ]:


# for i in ims:
#     show_state(None, image=i[0])


# In[ ]:


np.save("AICAR.npy", ims)


# In[ ]:


print('saving rollout in sad_car_noises.gif, open that file to see our sad results.')


# In[ ]:


imageio.mimsave('./sad_car_noises.gif', [resize(im.squeeze(), (256, 256, 3)) for im in ims])

