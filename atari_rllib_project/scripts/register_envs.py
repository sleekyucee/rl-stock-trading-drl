#reguster_envs

from gymnasium.envs.registration import register
from ale_py import ALEInterface
import os

#register function
def register_atari_envs():
    rom_dir = os.environ.get("ALE_ROM_DIR", "/users/adgs898/.python_envs/atari/roms")
    if not os.path.exists(rom_dir):
        raise FileNotFoundError(f"ALE ROM directory not found: {rom_dir}")
    for rom in os.listdir(rom_dir):
        if rom.endswith(".bin"):
            game = rom[:-4]
            env_id = f"ALE/{game}-v5"
            try:
                register(
                    id=env_id,
                    entry_point="ale_py.gym:ALE",
                    kwargs={"game": game, "obs_type": "rgb", "frameskip": 1},
                    max_episode_steps=10000,
                    nondeterministic=False,
                )
            except Exception as e:
                if "already registered" not in str(e):
                    print(f"Error registering {env_id}: {e}")