#train_ppo

import os
import sys
import argparse
import time
import warnings
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from register_envs import register_atari_envs
from frame_stack import FrameStack
from utils import load_config, set_seed
from logger import Logger
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore", category=DeprecationWarning)

def atari_env_creator(env_name):
    def make_env(_):
        register_atari_envs()
        env = gym.make(
            env_name,
            frameskip=1,
            full_action_space=False,
            repeat_action_probability=0.0
        )
        env = AtariPreprocessing(
            env,
            frame_skip=4,
            noop_max=30,
            grayscale_obs=True,
            scale_obs=True,
            terminal_on_life_loss=False
        )
        env = FrameStack(env, num_stack=4)
        return env
    return make_env

def train_ppo(config_path):
    try:
        config = load_config(config_path)
        env_settings = config["env_settings"]
        train_settings = config["train_settings"]
        experiment_settings = config["experiment_settings"]
    except Exception as e:
        print(f"[ERROR] Config loading failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    set_seed(env_settings.get("seed", 42))

    use_wandb = experiment_settings.get("use_wandb", False)
    wandb_logger = None
    if use_wandb:
        try:
            if "WANDB_MODE" not in os.environ:
                os.environ["WANDB_MODE"] = "online"
            wandb_logger = Logger(
                experiment_name=experiment_settings["experiment_name"],
                project=experiment_settings.get("project", "atari_rllib")
            )
            print("WandB initialized in online mode", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[WARNING] WandB initialization failed: {e}", file=sys.stderr, flush=True)
            use_wandb = False

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            object_store_memory=2 * 1024 * 1024 * 1024,
            _temp_dir=os.path.expanduser("~/ray_temp"),
            num_cpus=train_settings.get("num_rollout_workers", 8) + 2,
            _system_config={
                "health_check_initial_delay_ms": 0,
                "health_check_period_ms": 10000,
                "health_check_failure_threshold": 10
            }
        )

    try:
        tune.register_env(env_settings["env_name"], atari_env_creator(env_settings["env_name"]))
    except Exception as e:
        if "already registered" not in str(e):
            print(f"[ERROR] Environment registration failed: {e}", file=sys.stderr, flush=True)
            ray.shutdown()
            sys.exit(1)

    try:
        algo_config = (
            PPOConfig()
            .environment(
                env=env_settings["env_name"],
                clip_rewards=True,
                clip_actions=True,
                env_config={
                    "max_episode_steps": env_settings.get("max_episode_steps", 10000)
                }
            )
            .framework("torch")
            .env_runners(
                num_env_runners=train_settings.get("num_rollout_workers", 8),
                rollout_fragment_length=train_settings.get("rollout_fragment_length", 1000),
                num_envs_per_env_runner=1,
                observation_filter="NoFilter",
            )
            .training(
                lr=train_settings.get("lr", 0.00025),
                gamma=train_settings.get("gamma", 0.99),
                lambda_=train_settings.get("lambda", 0.95),
                clip_param=train_settings.get("clip_param", 0.2),
                vf_loss_coeff=train_settings.get("vf_loss_coeff", 0.5),
                entropy_coeff=train_settings.get("entropy_coeff", 0.01),
                train_batch_size=train_settings.get("train_batch_size", 4000),
                minibatch_size=train_settings.get("minibatch_size", 512),
                num_sgd_iter=train_settings.get("num_sgd_iter", 10)
            )
            .rl_module(
                model_config={
                    "conv_filters": [
                        [16, [8, 8], 4],
                        [32, [4, 4], 2],
                        [256, [11, 11], 1]
                    ],
                    "framestack": True
                }
            )
            .resources(
                num_gpus=train_settings.get("num_gpus", 1),
                num_cpus_per_worker=1,
                num_gpus_per_worker=0.1 if train_settings.get("num_gpus", 1) > 0 else 0
            )
            .reporting(
                min_sample_timesteps_per_iteration=1000,
                min_time_s_per_iteration=1
            )
        )

        algo = algo_config.build()
    except Exception as e:
        print(f"[ERROR] PPO build failed: {e}", file=sys.stderr, flush=True)
        ray.shutdown()
        sys.exit(1)

    max_iters = train_settings.get("max_iters", 1000)
    best_reward = float("-inf")

    models_dir = os.path.abspath("models")
    os.makedirs(models_dir, exist_ok=True)
    best_ckpt_dir = os.path.join(models_dir, "best_checkpoint")
    os.makedirs(best_ckpt_dir, exist_ok=True)

    tensorboard_dir = os.path.join("runs", experiment_settings["experiment_name"])
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    print(f"Starting PPO training for {max_iters} iterations...", file=sys.stderr, flush=True)
    training_start_time = time.time()

    for i in range(max_iters):
        print(f"Iteration {i + 1}/{max_iters}...", file=sys.stderr, flush=True)

        try:
            start_time = time.time()
            result = algo.train()
            duration = time.time() - start_time

            if duration > 300:
                print(f"Warning: Training iteration took too long ({duration:.2f}s)", file=sys.stderr, flush=True)

            episode_stats = result.get("env_runners", {})
            reward = episode_stats.get("episode_return_mean", -999)
            min_reward = episode_stats.get("episode_return_min", -999)
            max_reward = episode_stats.get("episode_return_max", -999)
            mean_ep_len = episode_stats.get("episode_len_mean", 0)
            episodes = episode_stats.get("num_episodes", 0)
            total_steps = result.get("num_env_steps_sampled_lifetime", 0)

            learner_stats = result.get("info", {}).get("learner", {}).get("default_policy", {})
            policy_loss = learner_stats.get("policy_loss", 0)
            vf_loss = learner_stats.get("vf_loss", 0)
            entropy = learner_stats.get("entropy", 0)
            kl = learner_stats.get("kl", 0)

            print(
                f"Iter {i + 1}: Reward={reward:.2f} (Min={min_reward:.2f}, Max={max_reward:.2f}) | "
                f"Steps={total_steps} | Episodes={episodes} | "
                f"Policy Loss={policy_loss:.4f} | VF Loss={vf_loss:.4f} | Entropy={entropy:.4f}",
                file=sys.stderr, flush=True
            )

            metrics = {
                "Reward/Mean": reward,
                "Reward/Min": min_reward,
                "Reward/Max": max_reward,
                "Timesteps/Total": total_steps,
                "Episode/Length": mean_ep_len,
                "Episode/Count": episodes,
                "Loss/Policy": policy_loss,
                "Loss/VF": vf_loss,
                "Entropy": entropy,
                "KL": kl
            }

            for metric, value in metrics.items():
                writer.add_scalar(metric, value, i + 1)
            writer.flush()

            if use_wandb and wandb_logger:
                wandb_logger.log({
                    "iteration": i + 1,
                    **metrics,
                    "timesteps_per_iteration": result.get("timesteps_this_iter", 0)
                })

            if reward > best_reward:
                best_reward = reward
                try:
                    checkpoint = algo.save(best_ckpt_dir)
                    print(f"New best checkpoint: {checkpoint.checkpoint.path}", file=sys.stderr, flush=True)

                    if use_wandb and wandb_logger:
                        wandb_logger.log({
                            "best_checkpoint_path": checkpoint.checkpoint.path,
                            "best_reward": best_reward
                        })
                except Exception as e:
                    print(f"Checkpoint failed: {e}", file=sys.stderr, flush=True)

        except Exception as e:
            print(f"Training failed: {e}", file=sys.stderr, flush=True)
            try:
                algo.restore(best_ckpt_dir)
                print("Restored from best checkpoint", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"Restore failed: {e}", file=sys.stderr, flush=True)
                break

    training_end_time = time.time()
    print(
        f"Training completed in {training_end_time - training_start_time:.2f}s. "
        f"Best reward: {best_reward:.2f}",
        file=sys.stderr, flush=True
    )

    writer.close()
    if use_wandb and wandb_logger:
        try:
            wandb_logger.finish()
        except Exception as e:
            print(f"WandB finish failed: {e}", file=sys.stderr, flush=True)

    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    try:
        register_atari_envs()
    except Exception as e:
        print(f"Environment registration warning: {e}", file=sys.stderr, flush=True)

    train_ppo(args.config)
