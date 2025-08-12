# Environment settings
ENV_NAME = "MountainCar-v0"
RENDER_MODE = "human"
SEED = 42

# Training parameters
NUM_EPISODES = 100

# FQL Agent parameters
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.1
NUM_ACTIONS = 3  # 0=left, 1=neutral, 2=right

# Fuzzy sets for position
POSITION_SETS = [
    (-1.2, -1.2, -0.6, -0.2),   # Low position
    (-0.6, -0.2, 0.2, 0.6),     # Medium position  
    (0.2, 0.6, 0.6, 0.6)        # High position
]

# Fuzzy sets for velocity
VELOCITY_SETS = [
    (-0.07, -0.07, -0.02, -0.005),   # Strong negative velocity
    (-0.02, -0.005, 0.005, 0.02),    # Near zero velocity
    (0.005, 0.02, 0.07, 0.07)        # Strong positive velocity
]

# Reward function parameters
REWARD_SUCCESS = 200         # Reward for reaching the goal
REWARD_POSITION_SCALE = 2.0  # Scale for position progress reward
REWARD_VELOCITY_RIGHT = 15.0 # Scale for rightward velocity reward
REWARD_VELOCITY_LEFT = 5.0   # Scale for leftward velocity reward (momentum)
REWARD_HEIGHT_SCALE = 15.0   # Scale for height bonus reward