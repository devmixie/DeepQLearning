__author__     = "Vijay Raj Franklin"
__license__    = "Apache License 2.0"
__version__    = "1.0.0"
__maintainer__ = "Vijay Raj Franklin"
__email__      = "franklynece@gmail.com"

import logging
logger = logging.getLogger(__name__)

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
logger.info("Num GPUs Available: %s ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam,SGD
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
        self.model         = self.build()
        self.model.summary();

        # check if a valid saved model file is present, if present load the weights from the file.
        if self.modelfl:
            self.load(os.path.join(self.modeldir,modelfl))
    
    def build_learning_rate(self):
        '''
            Builds the learning rate function to be used for learning.
        ''' 
        initial_learning_rate = 1e-5
        maximal_learning_rate = 1e-3
        step_size             = 200
        self.learning_rate = tfa.optimizers.ExponentialCyclicalLearningRate(
            initial_learning_rate,
            maximal_learning_rate,
            step_size
        )

    def build(self):
        model = tf.keras.applications.MobileNetV2(weights=None,input_shape=self.input_dim,classes=self.output_dim,classifier_activation="linear",pooling="avg")
        model.compile(loss='mse',optimizer=tfa.optimizers.AdamW(learning_rate=self.learning_rate,weight_decay=1e-4))
        
        return model;

    def load(self, fl):
        self.model.load_weights(fl)
        logger.info("Loaded model weights from file : %s", fl)

    def save(self, fl):
        self.model.save_weights(fl)
        logger.info("Saved model weights to file : %s", fl)