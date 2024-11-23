import os
import yaml
import torch
import shutil

class ExperimentManager:
    def __init__(self, config, experiment_name):
        self.config = config
        self.experiment_name = experiment_name
        self.base_dir = 'experiments/runs'
        self.experiment_dir = os.path.join(self.base_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')

    def setup_experiment(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def save_checkpoint(self, model, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None # Add optimizer state if needed
        }, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')

    def save_config(self, config_path):
         # Copy the config file to the experiment directory
        destination_path = os.path.join(self.experiment_dir, 'config.yaml')
        shutil.copyfile(config_path, destination_path)
        print(f'Config saved to {destination_path}')

    def log(self, message):
        log_path = os.path.join(self.log_dir, 'experiment.log')
        with open(log_path, 'a') as f:
            f.write(f'{message}\n')
        print(message)


    @property
    def parameters(self):
        # You can extend this later to get model parameters or other relevant info
        return self.config