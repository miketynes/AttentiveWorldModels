import tensorflow as tf
from tensorflow.keras.layers import (Dense, Input, Conv2D, 
                                     Conv2DTranspose,
                                     Reshape, Lambda, LSTM)
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
import numpy as np
import mdn
import utils

if tf.test.is_gpu_available():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# tf.keras.backend.set_floatx('float64')
    
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


def V(LATENT_SIZE=32, weights=None):
    """yeet to latent realm
    """
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

def V_inverse(LATENT_SIZE=32, weights=None):
    """Yeet's you back from the latent dimension.
    """
    latent_inputs = Input(shape=(LATENT_SIZE,), name='decoder_input')
    h = Dense(4*256, name="dec_fc")(latent_inputs)
    h = Reshape([1, 1, 4*256])(h)
    h = Conv2DTranspose(128, 5, strides=2, activation="relu", name="dec_deconv1")(h)
    h = Conv2DTranspose(64, 5, strides=2, activation="relu", name="dec_deconv2")(h)
    h = Conv2DTranspose(32, 6, strides=2, activation="relu", name="dec_deconv3")(h)
    outputs = Conv2DTranspose(3, 6, strides=2, activation='sigmoid', name="dec_deconv4")(h)

    decoder = Model(latent_inputs, outputs, name='decoder')
    if weights: 
        decoder.load_weights(weights)
    return decoder

def M(seq_len=128, act_len=3, output_dims=32, n_mixes=5, weights=None):
    M = Sequential([
        Input((None, act_len + utils.LATENT_SIZE)),
        LSTM(256, return_sequences=True),
        mdn.MDN(output_dims, n_mixes)
    ])

    M.compile(loss=mdn.get_mixture_loss_func(output_dims, n_mixes), 
              optimizer=tf.keras.optimizers.Adam(),
             )
    if weights: 
        M.load_weights(weights)
    return M

def C(_in=32+256, _out=3, weights=None):
    C = Controller(_in, _out)
    if weights:
        C.set_weights(np.load(weights))
    return C

def attn_M(weights=None):
    M = attention_mdn_rnn()
    if weights:
        M.load_weights(weights)
    return M

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(units, input_shape=(35,))
        self.W2 = tf.keras.layers.Dense(units, input_shape=(256,))
        self.V = tf.keras.layers.Dense(1, input_shape=(256,))

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        # hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # 
        features = tf.expand_dims(features, 0)
        hidden = tf.expand_dims(hidden, 0)
        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        # context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
   
    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({
            'units':self.units
        })
        return config
    
    def from_config(cls, config):
        return cls(**config)
    

class attention_mdn_rnn(tf.keras.Model):
    def __init__(self, 
                seq_len=128, 
                act_len=3, 
                latent_size=32, 
                cells=256, 
                output_dim=32, 
                n_mixes=5):
        super(attention_mdn_rnn, self).__init__()

        
        self.seq_len=seq_len
        self.act_len=act_len
        self.latent_size=latent_size
        self.cells=cells
        self.output_dim=output_dim
        self.n_mixes=n_mixes
        
        #self.inputs = Input((None, self.act_len + self.latent_size))
        self.lstm   = LSTM(self.cells,
                            return_sequences=True,
                            return_state=True,
                            recurrent_initializer='glorot_uniform')

        self.attention = BahdanauAttention(self.cells)
        self.out       = mdn.MDN(self.output_dim, self.n_mixes)
        
    def call(self, x, hidden):

        context_vector, attention_weights = self.attention(x, hidden)
        #context_vector = context_vector.numpy().squeeze()
        
        # context_vector = features * attention_weights

        x, hidden_out, c = self.lstm(context_vector[0])
        x = self.out(x)
        
        return x, hidden_out#, attention_weights

    
    def get_config(self):
        config = super(attention_mdn_rnn, self).get_config()
        config.update({'seq_len':self.seq_len,
                        'act_len':self.act_len,
                        'latent_size':self.latent_size,
                        'cells':self.cells,
                        'output_dim':self.output_dim,
                        'n_mixes':self.n_mixes})
        return config

    def from_config(cls, config):
        return cls(**config)
    
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.cells))


class Controller():
    def __init__(self, input_size, output_size):
        self._in = input_size
        self._out = output_size
        self.W = np.random.randn(input_size, output_size)
    
    def clip(self, x, lo=0.0, hi=1.0):
        return np.minimum(np.maximum(x, lo), hi)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __call__(self, obs):
        action = np.dot(obs, self.W)
        
        action[0] = np.tanh(action[0])
        action[1] = self.sigmoid(action[1])
        action[2] = self.clip(np.tanh(action[2]))
        
        return action
    
    def set_weights(self, W):
        # assume W is flat.
        self.W = np.reshape(W, self.W.shape)
        
    def randomly_init(self):
        self.W = np.random.randn(*self.W.shape)
       
    @property
    def shape(self):
        return self.W.shape
