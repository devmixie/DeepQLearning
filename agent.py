__author__     = "Vijay Raj Franklin"
__license__    = "Apache License 2.0"
__version__    = "1.0.0"
__maintainer__ = "Vijay Raj Franklin"
__email__      = "franklynece@gmail.com"

import logging
logger = logging.getLogger(__name__)

import numpy as np
import random

_RAND_SEED = 0xC0C0CAFE     # sets the random seed, for repeatability of observations.
np.random.seed(_RAND_SEED)
random.seed(_RAND_SEED)

from collections import deque

class Agent():
    def __init__(self, model, actions, train, lookback):
        self.model               = model
        self.actions             = actions
        self.action_space        = len(actions)
        self.train               = train
        self.lookback            = lookback
        logger.info("The game has the following actions [{}]".format(' '.join(map(str, actions))));
        #
        self.future_reward_discount_factor = 0.9    # factor which weighs down future reward estimation of our model.
        self.random_action_fallback_factor = 0.95   # factor which controls how frequent we choose a random action over our model's prection.
        self.random_action_decay_factor    = 0.975  # factor which controls how much we decay the random action fallover factor over time.
        self.random_action_factor_floor    = 0.01   # the minimum value to which the random_action_fallback_factor can decay to.
        if not self.train:
            self.random_action_fallback_factor = 0  # disable fallback to topk incase we are not training.
        self.skip_training_factor          = 0.5;
        #
        self.historical_space = deque(maxlen=1000)

    def next_move(self, state, max_topk_index):
        '''
            The function which takes in the input as the state of the game environment and provides the projected reward for all possible actions.
            Then the action which has the most projected reward is taken.
            Note that in order for the machine to explore new possibilities, at certain times a random action is taken.
            This way of picking a random action helps the machine to learn new action sequences which it had not learnt before - potentially.
            Usually this is done by choosing just a random action, but here we choose a topk-random action; that is choose a non-optimal solution from the topk results.
        '''
        #if state.shape[3] < self.lookback: # if we donot have a large enough state space covering the lookback, just return a random action. 
        if (state.shape[3] < self.lookback) or (np.random.rand() <= self.random_action_fallback_factor):
            random_action_idx = np.random.randint(0, len(self.actions))
            return self.actions[random_action_idx] # early exit
        
        # ###
        projected_rewards        = self.model.predict(state)            # predict the potential reward for all possible action.
        print ("projected_rewards ", projected_rewards, " epsilon ", self.random_action_fallback_factor)
        action_idx_at_max_reward = np.argmax(projected_rewards[0])      # choose the action that is projected to have the max reward.
        print ("projected_rewards ", projected_rewards, " epsilon ", self.random_action_fallback_factor, " action ", action_idx_at_max_reward)
        return  self.actions[action_idx_at_max_reward]

    def add_to_historical_space(self, state, action, reward, next_state, game_over):
        '''
            Adds to the known and already encountered historical spaces.
            state:      the current state of the game enviroment
            action:     the next action which the agent proposed to take
            reward:     the reward which was observed in the game moving from state with action.
            next_state: the actual next state which was observed in game after taking the action
            game_over:  was the next state and end state (game over etc..)
        '''
        if state.shape[3] >= self.lookback: #start adding to historical space when we have enough lookback. we stack states along 3rd axis, so checking the 3rd dimension for lookback
            self.historical_space.append((state, action, reward, next_state, game_over))

    def trainloop(self, batch_size):
        '''
            The taining function which adjusts the model's weight to perform better.
            The training process is done as follows:
                - select a sample of previously seen historical space, and use it for training input.
                - Calculate the lookahead reward. The look ahead reward is the reward that is expected in the next state that was seen.
                    - case 1: in case the next state is the end, the lookahead reward is the actual reward seen
                    - case 2: in case the next state is a valid state, use the model to predict the reward to be expected in next state.
                              note that, since this is in one step look ahead, use a discount factor to on lookahead prediction to account for uncertainity in future prediction.
                - calculate the current state reward vector as predicted by the model for all possible actions.
                - for the current state reward vector, change the reward in the direction of the action that was taken to the lookahead reward.
        '''
        if (np.random.rand() <= self.skip_training_factor):
            return;
        batch = random.sample(self.historical_space, batch_size)
        for state, action, reward, next_state, game_over in batch:
            lookahead_reward = reward
            if not game_over: # if the next state is not over, estimate what the model expects as reward in next state.
                lookahead_reward             = np.amax(self.model.predict(next_state)[0])
                discounted_lookahead_reward  = self.future_reward_discount_factor * lookahead_reward
                lookahead_reward             = (reward + discounted_lookahead_reward)
            # ###
            predicted_current_state_reward                = self.model.predict(state)
            action_idx                                    = self.actions.index(action)
            predicted_current_state_reward[0][action_idx] = lookahead_reward
            self.model.fit(state, predicted_current_state_reward, epochs=1, verbose=0)
        
        # ###
        if self.random_action_fallback_factor > self.random_action_factor_floor:
            self.random_action_fallback_factor *= self.random_action_decay_factor