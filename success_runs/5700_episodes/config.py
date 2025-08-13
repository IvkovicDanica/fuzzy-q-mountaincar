ENV_NAME = "CartPole-v1"
NUM_ACTIONS = 2

POSITION_SETS = [
    (-0.6, -0.6, -0.2,  0.0),   
    (-0.2, -0.05, 0.05, 0.2),   
    (0.0, 0.2, 0.6, 0.6)        
]
VELOCITY_SETS = [
    (-5.0, -5.0, -1.0, 0.0),
    (-1.0, -0.3, 0.3, 1.0),
    (0.0, 1.0, 5.0, 5.0)
]

GAMMA = 0.99
ALPHA = 0.005
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01

NUM_EPISODES = 10_000
MAX_STEPS_PER_EPISODE = 500  
SEED = 42 
