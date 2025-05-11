#train

import os
import sys
import ray
import argparse
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.logger import pretty_print
from scripts.utils import load_config, set_seed
from scripts.logger import Logger


def train_dqn(config_path):
    config = load_config(config_path)

    #set seed for reproducibility
    set_seed(config["env_settings"].get("seed", 42))

    #wandb logger setup
    use_wandb = config["experiment_settings"].get("use_wandb", True)
    experiment_name = config["experiment_settings"]["experiment_name"]
    wandb_logger = Logger(
        experiment_name=experiment_name,
        project=config["experiment_settings"].get("project", "atari_rllib")
    ) if use_wandb else None

    #init Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    #rllib config
    train_cfg = config["train_settings"]
    env_cfg = config["env_settings"]

    algo_config = (
        DQNConfig()
        .environment(env=env_cfg["env_name"])
        .framework("torch")
        .rollouts(num_rollout_workers=train_cfg["num_rollout_workers"])
        .training(
            lr=train_cfg["lr"],
            gamma=train_cfg["gamma"],
            train_batch_size=train_cfg["train_batch_size"],
            dueling=train_cfg["dueling"],
            double_q=train_cfg["double_q"]
        )
        .resources(num_gpus=train_cfg["num_gpus"])
    )

    algo = algo_config.build()
    max_iters = train_cfg["max_iters"]

    #track best model by reward
    best_reward = float("-inf")
    best_checkpoint_path = os.path.join("models", f"best_{experiment_name}")

    os.makedirs("models", exist_ok=True)

    for iteration in range(max_iters):
        result = algo.train()
        reward = result["episode_reward_mean"]

        print(pretty_print(result))

        if wandb_logger:
            wandb_logger.log({
                "iteration": iteration,
                "timesteps_total": result["timesteps_total"],
                "episode_reward_mean": reward,
                "episode_len_mean": result["episode_len_mean"],
                "exploration_epsilon": result["info"]
                    .get("learner", {})
                    .get("default_policy", {})
                    .get("cur_epsilon", None)
            })

        #save only best-performing checkpoint
        if reward > best_reward:
            best_reward = reward
            checkpoint_path = algo.save(best_checkpoint_path)
            print(f"New best checkpoint saved: {checkpoint_path} (Reward: {reward:.2f})")

            if wandb_logger:
                wandb_logger.save_file(checkpoint_path)

    if wandb_logger:
        wandb_logger.finish()

    ray.shutdown()
    print(f"\nTraining complete! Best reward: {best_reward:.2f}")
    print(f"Final best checkpoint saved to: {best_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    args = parser.parse_args()

    train_dqn(args.config)

