from dopamine.gyms.wiggle import Wiggle

def make_env(gym_name):
    '''Loads and builds the specified local gym.'''

    if gym_name == 'wiggle':
        return Wiggle()
