import torch
from torch.optim.lr_scheduler import StepLR


from training_library.runner import Runner
from training_library.modules.loss import multilayer_sisnr_loss
from training_library.models import SWave
from training_library.callbacks.splitbatch import SWaveSpliterCallback
from training_library.callbacks.clipgrad import ClipGradCallback
from training_library.data import Data
# from training_library.optim.optimizers import adam


def train():
    sr = 8000
    epochs = 6
    lr = 5e-4
    batch_size = 1
    tr_dir = "/mnt/disks/hdd0/datasets/speech_separation/dataset_8k/ds40k/tr"
    ev_dir = "/mnt/disks/hdd0/datasets/speech_separation/dataset_8k/ds40k/test"
    loss_func = multilayer_sisnr_loss

    model = SWave(N=128, L=8, H=128, R=6, C=2, sr=sr, segment=4,
                  input_normalize=True)

    train_data = Data.for_audio_separation(tr_dir, ev_dir, sr, batch_size)
    # optimizer = adam(model.parameters(), lr)
    # pytorch optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 betas=(0.9, 0.999))

    runner = Runner.create_standard_runner(model, train_data,
                                           loss_func, optimizer)

    # pytorch scheduler
    scheduler = StepLR(optimizer, 200, 0.98)
    runner.custom_scheduler(scheduler, every_iter=True)
    runner.remove_callback('standardsplitter')
    runner.add_callback([SWaveSpliterCallback(),
                         ClipGradCallback(max_grad=5)])

    training_configs = {"epochs": epochs,
                        "R": 6,
                        "sr": sr,
                        "optimizer": "adam_torch",
                        "bs": batch_size,
                        "dataset": "voxceleb1 mixture + WHAMR",
                        "dataset_version": 1,
                        "device": 0,
                        "train_size": len(train_data.train_dl),
                        "eval_size": len(train_data.valid_dl)
                        }

    runner.wandb_logger(training_configs, "speaker_separation_fb", "train_0")

    runner.add_csv_logger("train_log.csv")
    runner.fit(epochs)
    # runner.fit_exp(epochs)
    runner.recorder.plot_lr(save=True)
    runner.recorder.plot_train_loss(save=True)
    runner.recorder.plot_test_loss(save=True)
    runner.recorder.plot_momentum(save=True)
    runner.save(name="swave_e{}.pth".format(epochs))


if __name__ == "__main__":
    train()
