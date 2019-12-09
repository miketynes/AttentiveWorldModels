import os

from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from tensorflow.keras.callbacks import Callback

BATCH_SIZE = 128
LATENT_SIZE = 32

def preprocess_images(ims):
    ims = np.true_divide(ims, 255, dtype=np.float32)
    if len(ims.shape) == 3:
        ims = resize(ims, output_shape=(64, 64, 3))
        ims = np.expand_dims(ims, 0)
    else:
        ims = resize(ims, output_shape=(ims.shape[0], 64, 64, 3))
    return ims

def to_latent(model, ims):
    """This is bad. Encodes images one-batch at a time. 
    """
    assert(ims.shape[0] % 128 == 0)
    
    z = []
    for i, im_batch in enumerate(range(0, ims.shape[0], BATCH_SIZE)): 
        z.append(model(ims[BATCH_SIZE*i:BATCH_SIZE*(i+1)])[2].numpy().squeeze())
    return np.concatenate(z)


class TrainTimeCallback(Callback): 
    def __init__(self): 
        super(TrainTimeCallback, self).__init__()
        
    def on_train_begin(self, logs=None): 
        self._start_time = time()
        
    def on_train_end(self, logs=None):
        self._train_time = time() - self._start_time
        
    @property
    def train_time(self):
        s = self._train_time
        hours, remainder = divmod(s, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f'{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}'
    
    def print_train_time(self):
        print(f'Train time for model: {self.train_time}')
        
        
def load_folder(path, files):
    #files = os.listdir(path)
    _1 = np.load(os.path.join(path, files[0]))
    data = np.zeros((len(files), *_1.shape))
    for i, fname in enumerate(tqdm(files)):
        data[i] = np.load(os.path.join(path, fname))
    return data

