import tensorflow as tf
from tensorflow.keras.layers import (Dense, Input, Conv2D, 
                                     Reshape, Lambda, LSTM)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import mdn

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BATCH_SIZE = 128
LATENT_SIZE = 32

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


def Encoder(LATENT_SIZE=32, weights=None):
    inputs = Input(shape=(64, 64, 3), name='encoder_input')
    h = Conv2D(32, 4, strides=2, activation="relu", name="enc_conv1")(inputs)
    h = Conv2D(64, 4, strides=2, activation="relu", name="enc_conv2")(h)
    h = Conv2D(128, 4, strides=2, activation="relu", name="enc_conv3")(h)
    h = Conv2D(256, 4, strides=2, activation="relu", name="enc_conv4")(h)
    h = Reshape([2*2*256])(h)
    z_mean = Dense(LATENT_SIZE, name='z_mean')(h)
    z_log_var = Dense(LATENT_SIZE, name='z_log_var')(h)
    z = Lambda(sampling, output_shape=(LATENT_SIZE,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    if weights: 
        encoder.load_weights(weights)
    return encoder

def M(output_dims=32, n_mixes=5, weights=None)
    M = Sequential([
        Input((seq_len, act_len + utils.LATENT_SIZE)),
        LSTM(256, return_sequences=True),
        mdn.MDN(output_dims, n_mixes)
    ])

    M.compile(loss=mdn.get_mixture_loss_func(output_dims, n_mixes), 
              optimizer=tf.keras.optimizers.Adam(),
             )
    if weights: 
        M.load_weights(weights)
    return M
