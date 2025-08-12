import gymnasium as gym
import config
from src.fql_agent import FQLModel
from src.fuzzy_logic import Trapezium, InputStateVariable, Build

def train():

    position_sets = InputStateVariable(*[Trapezium(*p) for p in config.POSITION_SETS])
    velocity_sets = InputStateVariable(*[Trapezium(*v) for v in config.VELOCITY_SETS])
    fis = Build(position_sets, velocity_sets)

    # Create agent
    fql_agent = FQLModel(
        gamma=config.GAMMA,
        alpha=config.ALPHA,
        epsilon=config.EPSILON,
        action_set_length=config.NUM_ACTIONS,
        fis=fis
    )

    # Create environment
    env = gym.make(config.ENV_NAME, render_mode=config.RENDER_MODE)
    env.reset(seed=config.SEED)

    for episode in range(config.NUM_EPISODES):
        obs, info = env.reset()
        position, velocity = obs

        state = [position, velocity]
        action = fql_agent.get_initial_action(state)

        total_reward = 0

        for step in range(200):
            obs, base_reward, terminated, truncated, info = env.step(action)
            position, velocity = obs

            # Calculating reward
            reward = base_reward
            if position >= 0.5:
                reward += config.REWARD_SUCCESS

            reward += (position + 0.6) * config.REWARD_POSITION_SCALE

            if position > -0.5:
                reward += velocity * config.REWARD_VELOCITY_RIGHT
            else:
                reward += abs(velocity) * config.REWARD_VELOCITY_LEFT

            if position > 0.0:
                reward += position * config.REWARD_HEIGHT_SCALE

            total_reward += reward

            # Step
            next_state = [position, velocity]
            action = fql_agent.run(next_state, reward)

            if terminated or truncated:
                print(f"Episode {episode+1} finished after {step+1} steps, total reward: {total_reward:.2f}")
                break

    env.close()
    return fql_agent


def test(agent):
    env = gym.make(config.ENV_NAME, render_mode="human")
    obs, info = env.reset(seed=config.SEED)

    done = False
    while not done:
        position, velocity = obs
        state = [position, velocity]
        action = agent.get_action(state)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
