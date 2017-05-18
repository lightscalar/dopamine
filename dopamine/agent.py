import numpy as np


class Agent(object):
    '''Simple agent capable of navigating a given environment. 
    INPUTS
        brain - object
            A neural network model that will be used to make decisions.
        epsilon - float
            A number in [0,1] that controls balance between greedy/random 
            exploration. Agent will randomly select from among available 
            actions a fraction epsilon of the time.
   ''' 

    def __init__(self, brain, epsilon=0.2):
        self.brain = brain
        self.eps = epsilon


    def select_action(self, action_values):
        # Select the highest value action, or randomly select an action with
        # probability epsilon.
        if np.random.uniform() < self.eps:
            action = np.random.randint(len(action_values))
        else:
            action = np.argmax(action_values)
        return action


    def take_action(self, x):
        '''Given an observed input x, take action according to response from
        the brain.'''

        # Use the brain to predict action values.
        action_values = self.brain.predict(x)

        # Determine action to take.
        return int(self.select_action(action_values))
