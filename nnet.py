__author__     = "Vijay Raj Franklin"
__license__    = "Apache License 2.0"
__version__    = "1.0.0"
__maintainer__ = "Vijay Raj Franklin"
__email__      = "franklynece@gmail.com"

import logging
logger = logging.getLogger(__name__)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
logger.info("Num GPUs Available: %s ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa

class Nnet:
    '''
        Class that constructs the neural network which we use later for training.
        Note that the neural network is inherently seperated from the other parts of the code in separate file,
        it is usually benefitial to have this organized this way, so you can swap networks easily and explore.
    '''
    def __init__(self,input_dim, output_dim, modeldir, modelfl):
        self.input_dim     = input_dim      # input dimension for the neural network. 
        self.output_dim    = output_dim     # output dimension/space for the neural network.
        self.modeldir      = modeldir
        self.modelfl       = modelfl
        self.build_learning_rate()
        self.build()
        # check if a valid saved model file is present, if present load the weights from the file.
        if self.modelfl:
            self.load(os.path.join(self.modeldir,modelfl))
    
    def build_learning_rate(self):
        '''
            Builds the learning rate function to be used for learning.
        ''' 
        initial_learning_rate = 1e-6
        maximal_learning_rate = 1e-4
        step_size             = 30
        self.learning_rate = tfa.optimizers.ExponentialCyclicalLearningRate(
            initial_learning_rate,
            maximal_learning_rate,
            step_size
        )

    def build(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu',  kernel_initializer='he_uniform', padding='same', input_shape=self.input_dim))
        self.model.add(Conv2D(32, (3, 3), activation='relu',  kernel_initializer='he_uniform', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu',  kernel_initializer='he_uniform', padding='same'))
        self.model.add(Conv2D(64, (3, 3), activation='relu',  kernel_initializer='he_uniform', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(self.output_dim, activation="linear"))
        self.model.compile(loss='mse',optimizer=Adam(learning_rate=self.learning_rate))
        self.model.summary();

    def load(self, fl):
        self.model.load_weights(fl)
        logger.info("Loaded model weights from file : %s", fl)

    def save(self, fl):
        self.model.save_weights(fl)
        logger.info("Saved model weights to file : %s", fl)