__author__     = "Vijay Raj Franklin"
__license__    = "Apache License 2.0"
__version__    = "1.0.0"
__maintainer__ = "Vijay Raj Franklin"
__email__      = "franklynece@gmail.com"

import logging
logger = logging.getLogger(__name__)

import os

class Stats:
    '''
        An helper statistics class to keep track of how our training progresses.
        Helps to declutter the main training loop, keeping it clean and organized.
    '''
    def __init__(self, nnet, modeldir, train):
        self.score       = 0;        # counter that keeps track of the instantaneous performance of the agent.
        self.bestscore   = 0;        # counter to track the maximum performance reached by the agent.
        self.nnet        = nnet;     # direct handle to the nnet class so we can directly save the model from here.
        self.train       = train;    # flag that indicates if the framework is in training mode.
        self.modeldir    = modeldir; # directory to store the models.
    
    def handle_state(self,generation, game_over, scoredelta):
        '''
            increment the score by the score delta always making sure that we save the best performing model as we see a new best score.
        '''
        new_bestscore = False
        if not game_over:
            self.score += scoredelta       # increment the current score by the provided score delta.
            if self.bestscore < self.score:
                self.bestscore = self.score
                new_bestscore  = True
        else:
            self.score = 0;
        # ###
        logger.info("Generation %d last known best score %d current score %d", generation, self.bestscore, self.score)
        # ###
        if new_bestscore and self.train and game_over:
            logger.info("Reached new best score %d. Agent stored to disk.", self.bestscore)
            self.nnet.save(os.path.join(self.modeldir,"generation-"+ str(generation)+"-score-"+str(self.bestscore),"model"))
