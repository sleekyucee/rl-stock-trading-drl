#evaluate_agent

import os
import argparse
import ray
from ray.rllib.algorithms.dqn import DQNConfig
import gymnasium as gym

def evaluate(checkpoint_path, env_name="ALE/Pong-v5", num_episodes=3, render=True):
    ray.init(ignore_reinit_error=True)

    #build same config used in training
    config = (
        DQNConfig()
        .environment(env=env_name)
        .framework("torch")
        .rollouts(num_rollout_workers=0)  #no workers during evaluation
    )

    algo = config.build()
    algo.restore(checkpoint_path)

    env = gym.make(env_name, render_mode="human" if render else None)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done, total_reward = False, 0

        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep + 1} reward: {total_reward:.2f}")

    env.close()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to saved checkpoint")
    parser.add_argument("--env", default="ALE/Pong-v5", help="Gymnasium Atari env name")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes to run")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")

    args = parser.parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        env_name=args.env,
        num_episodes=args.episodes,
        render=not args.no_render
    )
