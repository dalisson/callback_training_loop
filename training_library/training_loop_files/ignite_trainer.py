# torch
import torch
from torch_lr_finder import LRFinder
from torch import optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingWarmRestarts, OneCycleLR
from torchvision import datasets, transforms
from torchsummary import summary
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Recall, Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping

# internal
from basetrainer import BaseTrainer
from loss_functions import AngularPenaltySMLoss
from softmax_dataloader import build_dataloaders
from data_exceptions import NotImplementedLoss, NotImplementedOptimizer, NotImplementedScheduler

#other
import numpy as np
import wandb
import os
import datetime
import time
import sys
from pathlib import Path
from tqdm import tqdm
from google.cloud import storage

# models
from neural_networks import TripletNet, DenseNet, SoftmaxDenseNet, ResNet, SoftmaxResNet, Flatten
from embeds import *
sys.path.insert(0, '/home/hulk/repos/vlad_network')
from network import VladNetwork

class IgniteTrainer(BaseTrainer):

    def __init__(self, gpu = True):
        super().__init__(gpu)

    def initialize_model(self, architecture):
        if architecture == "resnet":
            resnet_model = ResNet()
            self.model = SoftmaxResNet(resnet_model)
        elif architecture == "ghost_vlad":
            self.model = VladNetwork.load_keras_arch('/home/hulk/repos/vlad_network/vlad_network_from_kerasV1.pth')
        
    @staticmethod
    def _log_training_loss(trainer):
        print(f"Epoch: {trainer.state.epoch} - Training loss : {trainer.state.output}")

    def _log_training_results(self, trainer):
        self.evaluator.run(self.trainloader)
        metrics = self.evaluator.state.metrics
        wandb.log({"train_loss": metrics['loss'], "train_accuracy": metrics['accuracy'], "train_recall": metrics['recall']}, commit = False)
        print(f"Training metrics: Accuracy = {metrics['accuracy']} \ Recall = {metrics['recall']} \ Loss = {metrics['loss']}")

    def _log_evaluation_results(self, trainer):
        self.evaluator.run(self.testloader)
        metrics = self.evaluator.state.metrics
        wandb.log({"test_loss": metrics['loss'], "test_accuracy": metrics['accuracy'], "test_recall": metrics['recall']}, commit = True)
        print(f"Test metrics: Accuracy = {metrics['accuracy']} \ Recall = {metrics['recall']} \ Loss = {metrics['loss']}")

    def _scheduler_step(self, trainer):
        self.scheduler.step()

    def _log_learning_rate(self, trainer):
        print(f"Using LR = {self.scheduler.get_lr()}")

    def train(self, train_dir, test_dir, configs, wandb_project, wandb_name, keep_training_file = None):
        config = self._configure_wandb(configs, wandb_project, wandb_name)

        self.trainloader, self.testloader = build_dataloaders(train_dir, test_dir, configs["batch_size"], configs["data_augmentation"])
        self.initialize_model(configs["model"])

        params_to_train = self.model.parameters()
        if configs["tuning"] != "full_net":
            self.model.freeze()
            params_to_train = filter(lambda p: p.requires_grad, self.model.parameters())

        self._configure_criterion(configs["loss"], configs["embedding_size"], configs["n_speakers"])

        self._configure_optimizer(configs["optimizer"], params_to_train, configs["lr"], configs["weight_decay"])
        
        scheduler_configs = {"epochs": configs["epochs"], "max_lr": 0.001, "steps_per_epoch": len(self.trainloader)}
        self._configure_scheduler(configs["scheduler"], scheduler_configs)
    
        trainer = create_supervised_trainer(self.model, self.optimizer, self.criterion, device = self.device)
        self.evaluator = create_supervised_evaluator(self.model, device = self.device,
                                                metrics={
                                                    'accuracy': Accuracy(),
                                                    'recall': Recall(average = True),
                                                    'loss': Loss(self.criterion)
                                                    })

        checkpointer = ModelCheckpoint('ghost_vlad_models', 'without_nr', n_saved=2, save_as_state_dict=True, require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {f"{wandb_name}": self.model})
        trainer.add_event_handler(Events.ITERATION_STARTED(every = configs["log_interval"]), self._log_training_loss)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_training_results)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_evaluation_results)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self._scheduler_step)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_learning_rate)

        wandb.watch(self.model, log="all")

        trainer.run(self.trainloader, max_epochs=configs["epochs"])
