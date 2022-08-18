from ast import expr_context
from tabnanny import check
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import argparse

import os
import time

import dataloader as dataloader
from models import BaseMarioModel

class Module(pl.LightningModule):
    def __init__(self, data_path, model_hparams, optimizer_hparams):
        """
        Inputs:
            model_hparams - Hyperparameters for the model, as dictionary
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = BaseMarioModel(**model_hparams)
        self.loss_module = nn.MSELoss()
        self.data_path = data_path
        self.dataset_size = 0

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)

        return optimizer

    def training_step(self, batch, batch_idx):
        *input, labels = batch
        preds = self.model(*input)

        self.dataset_size += len(labels)
        
        labels = labels.reshape(-1, 1).float()
        labels = nn.functional.sigmoid(labels)
        loss = self.loss_module(preds, labels)

        # Log the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_mse', loss, on_step=False, on_epoch=True)
        return loss  # Return tensor with computational graph attached

    def training_epoch_end(self, training_step_outputs):
        self.log('dataset_size', self.dataset_size, on_epoch=True)
        self.dataset_size = 0

    def train_dataloader(self):
        while not os.path.exists(self.data_path):
            print('Training data not found, retrying in 10 seconds...')
            time.sleep(10)
        #During training dataloader function, create a new dataset with only a train split
        train_dataloader = None
        while not train_dataloader: 
            train_dataloader = dataloader.create_dataloader(
                    root=self.data_path, batch_size=args.batch_size, num_workers=args.num_workers, split=False)
        return train_dataloader

def train_model(checkpoint='', accelerator='gpu', devices=1, save_best=False, model_path='models', **kwargs):
    """
    Inputs:
        checkpoint - If given a checkpoint path, continue training from that point
        accelerator - Which accelerator to use ('cpu' or 'gpu')
        devices - How many devices to use
        model_path - Path to which to save the training data
        save_best - If true, save the model with the lowest training mse instead of the last
                    default False, because the dataloader is always changing
    """

    # Create a PyTorch Lightning trainer
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=model_path, log_graph=False, default_hp_metric=None)
    if save_best:
        callbacks = [ModelCheckpoint(save_weights_only=True, mode="min", monitor="train_mse")]
    else:
        callbacks = []
    trainer = pl.Trainer(default_root_dir=model_path, accelerator=accelerator, devices=devices,
                         log_every_n_steps=2, reload_dataloaders_every_n_epochs=2,
                         callbacks=callbacks, #Only save best model
                         logger=tb_logger)

    # Create and fit the model
    model = Module(**kwargs)
    if checkpoint:
        trainer.fit(model, ckpt_path=checkpoint)
    else:
        trainer.fit(model)
    model = Module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model

def get_next_input(module, input, deterministic=True):
    screenshot_tensor, previous_actions_tensor = dataloader.input_to_tensors(input["screenshots"], input["previous_points"])

    #Make batch of size 64 for screenshot tensor
    screenshot_tensor = torch.unsqueeze(screenshot_tensor, 0)
    screenshot_tensor = screenshot_tensor.expand(64, -1, -1, -1, -1)

    #Make batch of size 64, with possible actions, for action tensor
    previous_actions_tensor = torch.unsqueeze(previous_actions_tensor, 0)
    previous_actions_tensor = previous_actions_tensor.expand(64, -1, -1)

    possible_actions = []
    for i in range(64):
        #Try all possible actions using the A,B,X,Y buttons and one of the arrow buttons
        possible_actions.append(torch.tensor([int(i/32)%2==1, int(i/16)%2==1, int(i/8)%2==1, int(i/4)%2==1, i%4==0, i%4==1, i%4==2, i%4==3]))
    possible_actions_tensor = torch.stack(possible_actions)
    possible_actions_tensor = torch.unsqueeze(possible_actions_tensor, 1)
    actions_tensor = torch.cat((previous_actions_tensor, possible_actions_tensor), dim=1)
    
    #call model
    try:
        output = module.model(screenshot_tensor, actions_tensor)
    except RuntimeError as exc:
        print('Error during forward, probably caused by invalid input tensor sizes!')
        print('Screenshot tensor size:')
        print(screenshot_tensor.size())
        print('Action tensor size:')
        print(actions_tensor.size())
        raise ValueError("Invalid arguments during model forward") from exc
    if deterministic:
        output_index = torch.argmax(output.flatten()).item()
    else:
        output_index = torch.multinomial(output.flatten(), 1).item()
    return possible_actions[output_index].long().tolist()

def main(args):
    #Set seed for reproducability, use seed of 0 to not set a seed
    if not args.seed == 0:
        pl.seed_everything(args.seed)

    model_hparams = {"t": 4}

    os.makedirs(args.model_path, exist_ok=True)
    baseline_model, baseline_results = train_model(
            checkpoint=args.checkpoint,
            accelerator=args.accelerator, devices=args.devices,
            save_best=args.save_best,
            model_path = args.model_path,
            data_path = args.data_path,
            model_hparams=model_hparams,
            optimizer_hparams={"lr": 0.1})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Misc parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='seed used to use for setting the pseudo-random number generators, use 0 for no seed')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='If checkpoint string is set, resume training from that checkpoint')
    parser.add_argument('--save_best', action='store_true',
                        help='Save checkpoint with lowest test_mse score')
    parser.add_argument('--save_last', action='store_false', dest='save_best',
                        help='Save last checkpoint')
    parser.set_defaults(save_best=False)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size used in training and evaluation')
    parser.add_argument('--num_workers', type=int, default=2, #Higher crashes it for some reason, might want to look into it...
                        help='amount of workers for creating the dataloader')
    parser.add_argument('--model_path', type=str, default='models',
                        help='directory for saving models')
    parser.add_argument('--data_path', type=str, default='models',
                        help='directory where the data is stored')
    parser.add_argument('--accelerator', type=str, default='gpu',
                        help="which accelerator to use ('cpu' or 'gpu')")
    parser.add_argument('--devices', type=int, default=1,
                        help='How many devices to use')

    # Argument before training starts
    parser.add_argument('--sleep', type=int, default=0,
                        help='Sleep for certain amount of seconds before starting training')

    args = parser.parse_args()

    if args.sleep > 0:
        print("Training for mario_nn starting momentarily...")
        time.sleep(args.sleep)

    main(args)
