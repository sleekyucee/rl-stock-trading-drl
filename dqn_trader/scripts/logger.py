#logger

import wandb
import os

class Logger:
    def __init__(self, experiment_name, project="tsla_dqn"):
        """Initializes WandB for logging metrics."""
        self.logger = wandb.init(project=project, name=experiment_name)

    def log(self, metrics):
        """Logs metrics like reward, epsilon, etc."""
        self.logger.log(metrics)

    def save_file(self, file_path):
        """Logs a file to WandB using only filename (no symlink issues)."""
        try:
            filename_only = os.path.basename(file_path)
            wandb.save(filename_only)
            print(f"Logged to WandB: {filename_only}")
        except Exception as e:
            print(f"Skipped WandB save: {e}")

    def finish(self):
        """Closes the WandB run."""
        self.logger.finish()
