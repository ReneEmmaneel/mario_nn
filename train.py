import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import argparse

import os

import dataloader as dataloader
from models import BaseMarioModel

class Module(pl.LightningModule):
    def __init__(self, model_hparams, optimizer_hparams):
        """
        Inputs:
            model_hparams - Hyperparameters for the model, as dictionary
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = BaseMarioModel(**model_hparams)
        self.loss_module = nn.MSELoss()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)

        return optimizer

    def training_step(self, batch, batch_idx):
        *input, labels = batch

        preds = self.model(*input)
        

        loss = self.loss_module(preds, labels.reshape(-1, 1).float())

        # Log the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_loss', loss)
        return loss  # Return tensor with computational graph attached

    def validation_epoch_end(self, outputs):
        acc = torch.mean(torch.stack(outputs))

    def validation_step(self, batch, batch_idx):
        *input, labels = batch
        preds = self.model(*input)
        loss = self.loss_module(preds, labels.reshape(-1, 1).float())

        self.log('val_mse', loss)
    
        return loss

    def test_step(self, batch, batch_idx):
        *input, labels = batch
        preds = self.model(*input)
        loss = self.loss_module(preds, labels.reshape(-1, 1).float())

        self.log('test_mse', loss)

def train_model(train_dataloader, val_dataloader, test_dataloader, accelerator='gpu', devices=1, model_path='models', **kwargs):
    """
    Inputs:
        train_dataloader - DataLoader for the training examples
        val_dataloader - DataLoader for the evaluation examples
        test_dataloader - DataLoader for the test examples
        accelerator - Which accelerator to use ('cpu' or 'gpu')
        devices - How many devices to use
        model_path - Path to which to save the training data
    """

    # Create a PyTorch Lightning trainer
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=model_path, log_graph=False, default_hp_metric=None)
    trainer = pl.Trainer(default_root_dir=model_path, accelerator=accelerator, devices=devices,
                         log_every_n_steps=2,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_mse")], #Only save best model
                        logger=tb_logger)

    # Create and fit the model
    model = Module(**kwargs)
    trainer.fit(model, train_dataloader, val_dataloader)
    model = Module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    test_result = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    val_result = trainer.test(model, dataloaders=val_dataloader, verbose=False)

    return model, [test_result, val_result]

def main(args):
    #Set seed for reproducability, use seed of 0 to not set a seed
    if not args.seed == 0:
        pl.seed_everything(args.seed)

    #Load word embeddings, dataloaders and vocab from said dataloaders
    dataloaders = dataloader.create_dataloader(
            batch_size=args.batch_size, num_workers=args.num_workers)

    dataloader_train, dataloader_test, dataloader_evaluate = dataloaders

    model_hparams = {"t": 4}

    os.makedirs(args.model_path, exist_ok=True)
    baseline_model, baseline_results = train_model(
            dataloader_train, dataloader_evaluate, dataloader_test,
            accelerator=args.accelerator, devices=args.devices,
            model_path = args.model_path,
            model_hparams=model_hparams,
            optimizer_hparams={"lr": 0.1})

def get_next_input(module, input):
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
    output = module.model(screenshot_tensor, actions_tensor)
    best_index = torch.argmax(output, dim=0)
    return possible_actions[best_index].long().tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Misc parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='seed used to use for setting the pseudo-random number generators, use 0 for no seed')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size used in training and evaluation')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='amount of workers for creating the dataloader')
    parser.add_argument('--model_path', type=str, default='models',
                        help='directory for saving models')
    parser.add_argument('--accelerator', type=str, default='gpu',
                        help="which accelerator to use ('cpu' or 'gpu')")
    parser.add_argument('--devices', type=int, default=1,
                        help='How many devices to use')

    args = parser.parse_args()

    main(args)
