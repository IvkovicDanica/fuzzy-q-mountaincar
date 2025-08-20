# Fuzzy Q-Learning for CartPole Control

A reinforcement learning implementation that combines fuzzy logic with Q-learning to solve the CartPole-v1 environment. This project demonstrates how fuzzy inference systems can be integrated with traditional RL algorithms to handle continuous state spaces more effectively.

## Project Overview

This project implements a **Fuzzy Q-Learning (FQL)** agent that learns to balance a pole on a cart using fuzzy logic for state representation. Instead of discretizing the continuous state space, the agent uses trapezoidal membership functions to create fuzzy rules that naturally handle the continuous nature of the CartPole environment.

https://github.com/user-attachments/assets/3eaf8bca-288d-47c9-b8e2-b9505b10ff01

Example of the trained agent successfully balancing the pole after 1,251 episodes of training.
## Architecture

### Core Components

1. **FQLModel** (`src/fql_agent.py`): Main fuzzy Q-learning agent
   - Manages Q-table with dimensions (num_rules × num_actions)
   - Implements epsilon-greedy action selection weighted by rule activations
   - Updates Q-values using fuzzy rule memberships

2. **Fuzzy Logic System** (`src/fuzzy_logic.py`): 
   - `Trapezium`: Trapezoidal membership function implementation
   - `InputStateVariable`: Container for fuzzy sets per input variable
   - `Build`: Fuzzy inference system that computes rule activations

3. **Trainer** (`src/trainer.py`): Training loop with adaptive parameters and visualization

### Fuzzy Rule Generation

The system automatically generates **9 rules** from the combination of:
- **Pole Angle**: 3 fuzzy sets (Left, Center, Right)
- **Angular Velocity**: 3 fuzzy sets (Negative, Zero, Positive)

Each rule represents a fuzzy state-action mapping, allowing smooth transitions between different control strategies.

## Performance Results

The agent has achieved successful training runs across multiple episodes:

| Training Run | Episodes to Solution | Performance |
|-------------|---------------------|-------------|
| Run 5 | 1,251 episodes | ✅ Solved |
| Run 4 | 1,659 episodes | ✅ Solved |
| Run 3 | 2,700 episodes | ✅ Solved |
| Run 2 | 3,500 episodes | ✅ Solved |
| Run 1 | 5,700 episodes | ✅ Solved |

*CartPole-v1 is considered "solved" when the agent achieves an average reward of ≥475 over 100 consecutive episodes.*

## Getting Started

### Prerequisites

```bash
pip install numpy gymnasium scikit-fuzzy matplotlib
```

### Project Structure

```
fuzzy_q_cartpole/
├── config.py              # Hyperparameters and fuzzy set definitions
├── main.py                 # Entry point
├── src/
│   ├── fql_agent.py       # Fuzzy Q-Learning implementation
│   ├── fuzzy_logic.py     # Fuzzy inference system
│   └── trainer.py         # Training loop and evaluation
└── success_runs/          # Saved models and visualizations
    ├── 1251 episodes/
    ├── 1659 episodes/
    ├── 2700_episodes/
    ├── 3500_episodes/
    └── 5700_episodes/
```

### Configuration

Key parameters in `config.py`:

```python
# Environment
ENV_NAME = "CartPole-v1"
NUM_ACTIONS = 2

# Fuzzy Sets (left, left_top, right_top, right)
POSITION_SETS = [
    (-0.6, -0.6, -0.2,  0.0),  # Left
    (-0.2, -0.05, 0.05, 0.2),  # Center  
    (0.0, 0.2, 0.6, 0.6)       # Right
]

VELOCITY_SETS = [
    (-5.0, -5.0, -1.0, 0.0),   # Negative
    (-1.0, -0.3, 0.3, 1.0),    # Zero
    (0.0, 1.0, 5.0, 5.0)       # Positive
]

# Learning Parameters
GAMMA = 0.99          # Discount factor
ALPHA = 0.1           # Learning rate
EPSILON = 0.8         # Exploration rate
NUM_EPISODES = 10_000 # Maximum training episodes
```

### Training

```bash
python main.py
```

The trainer will:
- Initialize the fuzzy Q-learning agent
- Train for up to 10,000 episodes (or until solved)
- Display progress every 10 episodes
- Show live rendering every 500 episodes
- Save the best model when performance targets are met

### Success Run Analysis

Each successful run includes:
- `q_table_best_ep*.npy`: Saved Q-table weights
- `reward_per_episode.png`: Episode reward progression
- `moving_avg.png`: 100-episode moving average
- `q_table_stats.png`: Q-value distribution analysis
- `training.ipynb`: Jupyter notebook with detailed analysis
