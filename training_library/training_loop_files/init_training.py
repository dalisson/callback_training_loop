import wandb
#from ignite_trainer import IgniteTrainer
from basetrainer import BaseTrainer
from vlad_trainer import VladTrainer
#from v1_trainer import V1Trainer
from softmax_dataloader import build_dataloaders

if __name__ == '__main__':
    train_dir = "/home/hulk/datasets/preprocessed_datasets_from_db/1k_speakers_without_nr_257x250/train"
    test_dir = "/home/hulk/datasets/preprocessed_datasets_from_db/1k_speakers_without_nr_257x250/test"

    configs = {}

    # Configs to describe dataset and log on WandB
    configs["log_interval"] = 20
    configs["img_dims"] = (257, 250)
    configs["audio_format"] = "mp3_8k"
    configs["spectrogram_format"] = "16k"
    configs["preprocessing"] = "spectrogram"
    configs["transfer_learning"] = True
    configs["n_speakers"] = 1162
    configs["noise_reduction"] = False
    configs["model_prefix"] = "2.13"
    
    # Configs to modify train
    configs["batch_size"] = 32         
    configs["epochs"] = 30
    configs["lr"] = 1e-5
    configs["max_lr"] = 0.001
    configs["momentum"] = 0.9           
    configs["weight_decay"] = 5e-4
    configs["optimizer"] = "SGD"
    configs["scheduler"] = "one_cycle"
    configs["model"] = "ghost_vlad"
    configs["embedding_size"] = 512
    configs["data_augmentation"] = True
    configs["loss"] = "cosface"
    configs["tuning"] = "full_net"
    configs["gpu"] = True
    configs["keep_training_file"] = "ghost_vlad_models/2.13_e0.pth"
    #configs["keep_training_file"] = None

    trainer = VladTrainer(gpu = configs["gpu"])

    trainloader, testloader = build_dataloaders(train_dir, test_dir, configs["batch_size"], data_augmentation = configs["data_augmentation"])

    trainer.train(
        trainloader, 
        testloader,
        configs = configs,
        wandb_name=configs["model_prefix"], 
        wandb_project="ghost_vlad_pytorch",
        keep_training_file = configs["keep_training_file"],
    )
