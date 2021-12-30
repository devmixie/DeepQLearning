__author__     = "Vijay Raj Franklin"
__license__    = "Apache License 2.0"
__version__    = "1.0.0"
__maintainer__ = "Vijay Raj Franklin"
__email__      = "franklynece@gmail.com"

import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
from skimage.transform import resize

#let's import the pygame learning environment (hereafter referred to as ple) and our game.
from ple import PLE
from ple.games.pixelcopter import Pixelcopter 

#let's import our configuration helper, neural network and the decision agent.
from extractconfig import ConfigParser
from nnet import Nnet
from agent import Agent
from stats import Stats
from collections import deque

class Game:
    def __init__(self,cfgjson,modeldir,modelfl):
        '''
            The initialization method which sets up the framework.
            The framework is setup in steps as follows:
                step 1: configuration file parsing setup
                step 2: create a game object
                step 3: create & build our neural network
                step 4: create our agent which uses out neural network to control the game
                step 5: initialize the statistics class to observe and record the stats.
                step 6: run loop to progressively train and refine the neural network.
        '''
        self.train    = True if not modelfl else False
        if self.train:
            logger.info("Activated online training mode for the agent")
        else:
            logger.info("Activated pretrained agent load mode")
        # ###
        self.cfg      = ConfigParser(cfgjson).extract() # extract the configs from the json file.

        self.build_game() #build the game environment

        # _nnet_input_dim:
        # the input dimension for the neural net is the environment state on which we would like your nnet to operate on.
        # in this case, we are learning to map the screen capture to an expected reward, so the input is the screen capture image
        # The four components in the input dimension are defined as follows:
        #       None : this indicates that the first dimension is unspecified, this would be later used to pass multiple images in parallel (in batches)
        #            (ie) this is like a placeholder, where can use to pass varying number of images in parallel
        #       frame_height x frame_width x frame_channels: 
        #            dimension of the input image which we operate on.  
        _nnet_input_dim = (self.cfg['display']['display_width'],
                            self.cfg['display']['display_height'],
                            self.cfg['trainer']['lookback'])
        
        # _nnet_output_dim:
        # the output dimension for the neural net is basically the rewards for each of the possible actions.
        #   That is, there would be as many number of outputs as there are number of actions. 
        #   For example, if the system/game has two possible actions (say UP & DOWN), the nnet would have outputs at each instant,
        #   giving the estimated reward for each of the possible action.
        _nnet_output_dim      = len(self.environment.getActionSet())

        self.nnet             = Nnet(_nnet_input_dim,_nnet_output_dim,modeldir,modelfl)
        self.model            = self.nnet.model; #the actual neural network instance which is used for taking action in environment.
         
        # let's build our agent using our model
        self.agent            = Agent(self.model, self.environment.getActionSet(),self.train,self.cfg['trainer']['lookback'])
        self.max_topk_index   = max(self.cfg['runner']['max_topk_index'],len(self.environment.getActionSet()))

        # let's initialize our statistics model
        self.stats            = Stats(self.nnet,modeldir,self.train)

        # state variable holding the last lookback number of seen states
        self.state            = deque(maxlen=self.cfg['trainer']['lookback'])

    def build_game(self):
        '''
            Create an instance of the pygame learning environment with our game.
                Note that the display screen resolution is directly proportional to the computational complexity.
                Having a larger resolution is good for visual asthetics but, the model will converge better and faster with a lower resolution,
                especially in game screens like these, where there are not much detail to be gained with increased resolution.
         '''
        game            = Pixelcopter(self.cfg['display']['display_width'],self.cfg['display']['display_height'])
        # Note: render display controls if the game is to be displayed on screen.
        self.environment = PLE(game, fps=self.cfg['display']['frames_per_second'],display_screen=self.cfg['display']['render_display'])

    def run(self):
        def process_state(state):
            state = state.reshape(1,state.shape[0],state.shape[1],1)
            self.state.append(state)
            return np.concatenate(self.state,axis=3)

        # ###
        for generation in range(self.cfg['runner']['max_generation']):
            self.environment.init()
            reward    = 0;       # variable that tracks the reward that is got on every action
            game_over = False;   # variable that tracks if the game is still in progress or if it had ended (Ex: the actor crashed, timeout)
            state     = process_state(self.environment.getScreenGrayscale())
            for frame in range(self.cfg['runner']['max_frame_per_generation']):
                if self.environment.game_over():
                    self.environment.reset_game()
                # ###
                action     = self.agent.next_move(state,self.max_topk_index)
                reward     = self.environment.act(action)
                game_over  = self.environment.game_over()
                # ###
                if reward == 0: reward = self.cfg['runner']['score_delta_per_frame']   # make sure we do positive reinforcement, IF the game is still going on.
                if game_over:   reward = self.cfg['runner']['penalty_on_game_over']
                self.stats.handle_state(generation, game_over, self.cfg['runner']['score_delta_per_frame'])

                # ###
                next_state = process_state(self.environment.getScreenGrayscale())
                self.agent.add_to_historical_space(state, action, reward, next_state, game_over)
                if self.train and len(self.agent.historical_space) >= self.cfg['trainer']['batch_size']:
                    self.agent.trainloop(self.cfg['trainer']['batch_size'])
                state      = next_state;
                # ###
                