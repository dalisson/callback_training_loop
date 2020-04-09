import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingWarmRestarts, OneCycleLR
from torchsummary import summary
from ignite.metrics import Accuracy, Recall, Precision

import os
import sys
import wandb

from tqdm import tqdm
from pathlib import Path
from google.cloud import storage

from softmax_dataloader import build_dataloaders
from loss_functions import AngularPenaltySMLoss
from data_exceptions import NotImplementedLoss, NotImplementedOptimizer, NotImplementedScheduler

#inserindo outra pasta de onde vem a am_softmax_loss
sys.path.insert(1, '../vlad_network')
from am_softmax_loss import AMSoftmaxLoss
class BaseTrainer:
    def __init__(self, gpu: bool = True):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/hulk/minds-digital-api.json"
        
        if gpu:
            try:
                d = input("Train on GPU: ")
                torch.cuda.set_device(int(d))
                self.device = torch.cuda.current_device()
                print("\n-> Using GPU:", self.device)
            except:
                print("No GPU found, using CPU instead.")
                self.device = "cpu"
        else:
            self.device = "cpu"

    def initialize_model(self, verbose = True):
        '''
            Method to initialize a model architecture.
        '''
        if verbose:
            print("Initializing model architecture...")
        self.model = None
        self.model.to(self.device)

    def load_weights_from_file(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint)

    def load_optimizer_from_file(self, filepath):
        checkpoint = torch.load(filepath)
        self.optimizer.load_state_dict(checkpoint)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def load_scheduler_from_file(self, filepath):
        checkpoint = torch.load(filepath)
        self.scheduler.load_state_dict(checkpoint)

    def save_model_on_disk(self, state_dict, path: str):
        '''
            Method to save model state_dict on disk to the specified filepath.
            Args:
                state_dict: (dict) torch model state_dict containing weights/biases or a dictionary with multiple state_dicts (optimizer, scheduler, etc...)
                path: (string) path to save the .pth file.
        '''
        print("Saving model...")
        path = str(Path("ghost_vlad_models").joinpath(path))
        torch.save(state_dict, path)
        if self.config.loss in ["sphereface", "arcface", "cosface",'am_softmax']:
            torch.save(self.criterion.fc.state_dict(), f"am_softmax_layers/{self.config.model_prefix}_amsoftmax_fc.pth")
        print("Model saved on disk.")
    
    @staticmethod
    def save_model_on_storage(source_name, dest_name):
        '''
            Method to save model state dict to a file on Google Storage.
            Args:
                source_name: (string) path to the .pth file containing the state_dicts.
                dest_name: (string) Google Storage path to save the file.
        '''
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("minds-callcenter")
        blob = bucket.blob(dest_name)
        blob.upload_from_filename(source_name)
        print("Model saved on Google Storage")

    def _configure_wandb(self, configs, wandb_project, wandb_name, entity = "minds", verbose = True):
        if verbose:
            print(f"\n--> Configuring WandB run. Project name: {wandb_project}. Run name: {wandb_name}")
        wandb.init(name = wandb_name, project = wandb_project, entity = entity)
        config = wandb.config 
        for k, v in configs.items():
            setattr(config, k, v)
        return config

    def _configure_criterion(self, loss_name, embedding_size = None, n_speakers = None, verbose = True):
        if verbose:
            print(f"\n--> Configuring loss function... ({loss_name})")
        if loss_name == "cross-entropy":
            self.model = torch.nn.Sequential(self.model, torch.nn.Linear(embedding_size, n_speakers))
            self.criterion = torch.nn.CrossEntropyLoss()
            return list(self.model.parameters())
        elif loss_name in ["sphereface", "arcface", "cosface"]:
            self.criterion = AngularPenaltySMLoss(embedding_size, n_speakers, loss_type = loss_name).to(self.device)
            wandb.watch(self.criterion.fc)
            if self.config.keep_training_file:
                fc = torch.load(f"am_softmax_layers/{self.config.model_prefix}_amsoftmax_fc.pth")
                self.criterion.fc.load_state_dict(fc)
            return list(self.model.parameters()) + list(self.criterion.fc.parameters())
        elif loss_name == 'am_softmax':
            self.criterion = AMSoftmaxLoss(embedding_size, n_speakers).to(self.device)
            return list(self.model.parameters()) + list(self.criterion.parameters())
        else:
            raise NotImplementedLoss(loss_name)
        if self.config.keep_training_file:
            self.load_scheduler_from_file(f"ghost_vlad_models/scheduler_{self.config.model_prefix}.pth")
        return

    def _configure_optimizer(self, optim_name, params_to_train, lr, weight_decay, verbose = True):
        if verbose:
            print(f"\n--> Configuring optimizer... ({optim_name})")
        if optim_name == "AdamW":
            self.optimizer = torch.optim.AdamW(params_to_train, lr = lr, weight_decay=weight_decay)
        elif optim_name == "SGD":
            self.optimizer = torch.optim.SGD(params_to_train, lr = lr, weight_decay=weight_decay)
        elif optim_name == "Adam":
            self.optimizer = torch.optim.Adam(params_to_train, lr = lr, weight_decay=weight_decay)
        else:
            raise NotImplementedOptimizer(optim_name)
        if self.config.keep_training_file:
            self.load_optimizer_from_file(f"ghost_vlad_models/optimizer_{self.config.model_prefix}.pth")
        return 
    
    def _configure_scheduler(self, scheduler_name, configs = None, verbose = True):
        if verbose:
            print(f"\n--> Configuring learning rate scheduler... ({scheduler_name})")
        if scheduler_name == "one_cycle":
            self.scheduler = OneCycleLR(self.optimizer, max_lr = configs["max_lr"], steps_per_epoch = configs["steps_per_epoch"], epochs = configs["epochs"])
        elif scheduler_name == "exponential":
            self.scheduler = ExponentialLR(self.optimizer, gamma = configs["gamma"])
        elif scheduler_name == "step":
            self.scheduler = StepLR(self.optimizer, step_size = configs["step_size"])
        elif scheduler_name == "cosine_annealing":
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0 = configs["T_0"], T_mult = configs["T_mult"])
        else:
            raise NotImplementedScheduler(scheduler_name)
        return

    def basic_training_iteration(self, samples, targets):
        '''
            Method to run a basic PyTorch training iteration (getting model output for data, calculating loss, calculating gradients and
        performing optmizer step).
            Returns:
                Calculated metrics for that data.
        '''

        self.model.train()
        samples, targets = samples.to(self.device).float(), targets.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(samples)
        #output = torch.nn.functional.normalize(output, p = 2, dim = 1)

        if self.config.loss in ["sphereface", "arcface", "cosface", 'am_softmax']:
            self.criterion.fc.train()
            loss, output = self.criterion(output, targets)
        else:
            loss = self.criterion(output, targets)

        loss.backward()
        self.optimizer.step()

        self.accuracy.update((output, targets))
        self.recall.update((output, targets))
        self.precision.update((output, targets))

        self.scheduler.step()
        return loss

    def evaluate(self, testloader, reduction = "mean", verbose = True):
        if verbose:
            print("Evaluating model...")
        self.model.eval()
        with torch.no_grad():
            losses = []
            accuracy = Accuracy()
            recall = Recall(average = True)
            precision = Precision(average = True)
            for samples, targets in testloader:
                samples, targets = samples.to(self.device).float(), targets.to(self.device)
                output = self.model(samples)
                #output = torch.nn.functional.normalize(output, p = 2, dim = 1)
                
                if self.config.loss in ["sphereface", "arcface", "cosface", 'am_softmax']:
                    self.criterion.fc.eval()
                    loss, output = self.criterion(output, targets)
                else:
                    loss = self.criterion(output, targets)
                losses.append(loss)
                
                accuracy.update((output, targets))
                recall.update((output, targets))
                precision.update((output, targets))
                
            if verbose:
                print(f"Evaluation results: Loss = {torch.tensor(losses).mean().item()} \ Accuracy = {accuracy.compute()} \ Recall = {recall.compute()} \ Precision = {precision.compute()}")
            if reduction == "mean":
                results = {"loss": torch.tensor(losses).mean().item(), "accuracy": accuracy.compute(), "recall": recall.compute(), "precision": precision.compute()}
                return results
            elif reduction == "sum":
                results = {"loss": torch.tensor(losses).sum().item(), "accuracy": accuracy.compute(), "recall": recall.compute(), "precision": precision.compute()}
                return results
            else:
                results = {"loss": torch.tensor(losses), "accuracy": accuracy.compute(), "recall": recall.compute(), "precision": precision.compute()}
                return results

    def train(self, trainloader, testloader, configs: dict, wandb_project: str, wandb_name: str, evaluate_each_epoch = True, keep_training_file = None, verbose = True, log_on_wandb = True):
        '''
            Main method containing the training definitions (dataloaders, optimizer, loss, scheduler, loading model) and training loop.
            Args:
                trainloader: DataLoader object for the training data.
                testloader: DataLoader object for the testing data.
                configs: (dict) dictionary contaning all the configurations you may need to build the training session.
                    Required configs fields:
                        - batch_size: int
                        - epochs: int
                        - lr: float
                        - momentum: float
                        - weight_decay: float
                        - optimizer: string
                        - scheduler: string
                        - model: string
                        - embedding_size: int
                        - data_augmentation: bool
                        - loss: string
                        - tuning: string
                        - input_shape: tuple

                    Optional fields (strings, ints, floats, booleans, tuples):
                        - any field containing information about the run to send to wandb configuration run

                wandb_project: (string) name of the Weights & Biases project.
                wandb_name: (string) name of the training session that appears on Weights & Biases inside wandb_project.
                keep_training_file: (string) path to a saved model to continue the traning from.
                verbose: (bool) weather to print out running steps and results.
        '''

        self.config = self._configure_wandb(configs, wandb_project, wandb_name, verbose=verbose)

        self.initialize_model()
        if keep_training_file:
            self.load_weights_from_file(keep_training_file)
        if configs["tuning"] != "full_net":
            self.model.freeze()
        wandb.watch(self.model, log="all")

        if verbose:
            print(summary(self.model, (1, configs["img_dims"][0], configs["img_dims"][1])))
        
        params = self._configure_criterion(configs["loss"], configs["embedding_size"], configs["n_speakers"], verbose = verbose)

        self._configure_optimizer(configs["optimizer"], params, configs["lr"], configs["weight_decay"], verbose = verbose)
        
        scheduler_configs = {"epochs": configs["epochs"], "max_lr": configs["max_lr"], "steps_per_epoch": len(trainloader)}
        self._configure_scheduler(configs["scheduler"], scheduler_configs, verbose = verbose)
        self.model.to(self.device)

        self.accuracy = Accuracy()
        self.recall = Recall(average = True)
        self.precision = Precision(average = True)

        t = tqdm(range(self.config.epochs), "Training...", self.config.epochs)
        for e in t:
            t.set_description(f"Epochs {e}/{self.config.epochs}")
            
            train_losses = []
            for i, data in enumerate(trainloader):
                samples, targets = data[0], data[1]
                batch_loss = self.basic_training_iteration(samples, targets)
                train_losses.append(batch_loss)

                if verbose:
                    if i % configs["log_interval"] == 0:
                        print(f"Batch {i}/{len(trainloader)} - Loss = {batch_loss}")

           # self.save_model_on_disk(self.model.state_dict(), f"{configs['model_prefix']}_e{e}.pth")
           # self.save_model_on_disk(self.optimizer.state_dict(), f"optimizer_{configs['model_prefix']}.pth")
           # self.save_model_on_disk(self.scheduler.state_dict(), f"scheduler_{configs['model_prefix']}.pth")
            
            eval_results = None
            if evaluate_each_epoch:
                eval_results = self.evaluate(testloader, verbose = verbose)
            
            if log_on_wandb:
                wandb.log({"train_loss": torch.tensor(train_losses).mean().item(), "train_accuracy": self.accuracy.compute(), "train_recall": self.recall.compute(), "test_loss": eval_results["loss"], "test_accuracy": eval_results["accuracy"], "test_recall": eval_results["recall"]})

            if verbose:
                print(f"--> Using Learning Rate = {self.scheduler.get_lr()}")

            self.accuracy.reset()
            self.recall.reset()
            self.precision.reset()


