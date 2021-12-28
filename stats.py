__author__     = "Vijay Raj Franklin"
__license__    = "Apache License 2.0"
__version__    = "1.0.0"
__maintainer__ = "Vijay Raj Franklin"
__email__      = "franklynece@gmail.com"

import logging
logger = logging.getLogger(__name__)

import os
from collections import deque
from statistics import mean

class Stats:
    '''
        An helper statistics class to keep track of how our training progresses.
        Helps to declutter the main training loop, keeping it clean and organized.
    '''
    def __init__(self, nnet, modeldir, train):
        self.score         = 0.0;      # counter that keeps track of the instantaneous performance of the agent.
        self.bestscore     = 0.0;      # counter to track the maximum performance reached by the agent.
        self.nnet          = nnet;     # direct handle to the nnet class so we can directly save the model from here.
        self.train         = train;    # flag that indicates if the framework is in training mode.
        self.modeldir      = modeldir; # directory to store the models.
        self.score_history = deque(maxlen=20)
    
    def handle_state(self,generation, game_over, scoredelta):
        '''
            increment the score by the score delta always making sure that we save the best performing model as we see a new best score.
        '''
        if not game_over:
            self.score   += scoredelta       # increment the current score by the provided score delta.
        else:
            self.score   += 0.0;
        # ###
        if self.bestscore < self.score and self.train and game_over:
            self.bestscore = self.score
            logger.info("Reached new best score %d. Agent stored to disk.", self.bestscore)
            self.nnet.save(os.path.join(self.modeldir,"generation-"+ str(generation)+"-score-"+str(int(self.bestscore*10)),"model"))
        # ###
        if game_over:
            self.score_history.append(self.score) 
            logger.info("Game Over: Generation %d last known best score %d Last20 runs avg score %d current score %d", generation, self.bestscore*10, mean(self.score_history)*10, self.score*10)
            self.score = 0.0;