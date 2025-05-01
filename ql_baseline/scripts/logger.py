#scripts/logger.py

import wandb

class Logger:
    def __init__(self, experiment_name, project="tsla_qlearning"):
        """Initializes WandB for logging metrics and saving models."""
        self.logger = wandb.init(project=project, name=experiment_name)

    def log(self, metrics):
        """Logs metrics like reward, epsilon, etc."""
        self.logger.log(metrics)

    def save_file(self, file_path):
        """Saves a file (e.g., Q-table) to WandB."""
        wandb.save(file_path)

    def finish(self):
        """Closes the WandB run."""
        self.logger.finish()

