import numpy as np
from skimage.transform import resize

BATCH_SIZE = 128
LATENT_SIZE = 32

import tensorflow.keras.backend as K


def preprocess_images(ims):
    ims = np.true_divide(ims, 255, dtype=np.float32)
    ims = resize(ims, output_shape=(1024, 64, 64, 3))
    return ims

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    
    # import pdb; pdb.set_trace()
    z_mean, z_log_var = args
    #_batch = z_mean.shape[0]
    #_dim = z_mean.shape[1]
    batch = BATCH_SIZE# if _batch is None else _batch
    dim = LATENT_SIZE #if _dim is None else _dim
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    foo = z_mean + K.exp(0.5 * z_log_var)# * epsilon
    bar = foo * epsilon
    return bar


def to_latent(model, ims):
    """This is bad. Encodes images one-batch at a time. 
    """
    assert(ims.shape[0] % 128 == 0)
    
    z = []
    for i, im_batch in enumerate(range(0, ims.shape[0], BATCH_SIZE)): 
        z.append(model(ims[BATCH_SIZE*i:BATCH_SIZE*(i+1)])[2].numpy().squeeze())
    return np.concatenate(z)
