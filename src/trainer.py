import random
import numpy as np
import gymnasium as gym
import config
from src.fql_agent import FQLModel
from src.fuzzy_logic import Trapezium, InputStateVariable, Build
from collections import deque

def train():
    if getattr(config, "SEED", None) is not None:
        random.seed(config.SEED)
        np.random.seed(config.SEED)

    position_sets = InputStateVariable(*[Trapezium(*p) for p in config.POSITION_SETS])
    velocity_sets = InputStateVariable(*[Trapezium(*v) for v in config.VELOCITY_SETS])
    fis = Build(position_sets, velocity_sets)

    fql_agent = FQLModel(
        gamma=config.GAMMA,
        alpha=config.ALPHA,
        epsilon=config.EPSILON,
        action_set_length=config.NUM_ACTIONS,
        fis=fis
    )

    env = gym.make(config.ENV_NAME)
    episode_rewards = []
    ma_window = 100
    ma_deque = deque(maxlen=ma_window)
    best_ma = -np.inf

    for episode in range(1, config.NUM_EPISODES + 1):
        if getattr(config, "SEED", None) is not None:
            obs, _ = env.reset(seed=(config.SEED + episode))
        else:
            obs, _ = env.reset()

        # obs: (cart_pos, cart_vel, pole_angle, pole_ang_vel)
        cart_pos, cart_vel, pole_angle, pole_ang_vel = obs
        state = [float(pole_angle), float(pole_ang_vel)]

        action_index = fql_agent.get_initial_action(state)  # returns 0 or 1
        total_reward = 0.0
        terminated = truncated = False

        for step in range(config.MAX_STEPS_PER_EPISODE):
            obs, base_reward, terminated, truncated, info = env.step(int(action_index))
            cart_pos, cart_vel, pole_angle, pole_ang_vel = obs

            reward = base_reward
            TERMINATION_PENALTY = getattr(config, "TERMINATION_PENALTY",  -20.0)
            if terminated or truncated:
                reward += TERMINATION_PENALTY

            total_reward += reward

            next_state = [float(pole_angle), float(pole_ang_vel)]
            action_index = fql_agent.run(next_state, reward)

            if terminated or truncated:
                break

        fql_agent.epsilon = max(config.EPSILON_MIN, fql_agent.epsilon * config.EPSILON_DECAY)
        episode_rewards.append(total_reward)
        ma_deque.append(total_reward)
        moving_avg = float(np.mean(ma_deque))
        if moving_avg > best_ma:
            best_ma = moving_avg

        if episode % 10 == 0:
            print(f"Ep {episode} reward {total_reward:.1f} MA({ma_window}) {moving_avg:.2f} eps {fql_agent.epsilon:.3f}")
            
        if episode % 100 == 0:
            print("q_table stats: min {:.3f}, mean {:.3f}, max {:.3f}".format(
                np.min(fql_agent.q_table), np.mean(fql_agent.q_table), np.max(fql_agent.q_table)))
        
        if episode % 500 == 0:
            try:
                render_env = gym.make(config.ENV_NAME, render_mode="human")
                prev_eps = fql_agent.epsilon
                fql_agent.epsilon = 0.0

                obs, _ = render_env.reset(seed=(getattr(config, "SEED", None) or None))
                done = False
                while not done:
                    _, _, pole_angle, pole_ang_vel = obs
                    action = fql_agent.get_action([float(pole_angle), float(pole_ang_vel)])
                    obs, _, terminated, truncated, _ = render_env.step(int(action))
                    done = terminated or truncated

                fql_agent.epsilon = prev_eps
                render_env.close()
            except Exception as e:
                print("Render failed:", e)

        # early stopping: solved CartPole-v1 if 100-episode avg >= 475
        if episode >= ma_window and moving_avg >= 475.0:
            print(f"Solved at episode {episode} (moving avg {moving_avg:.2f})")
            break

    env.close()
    return fql_agent, episode_rewards
