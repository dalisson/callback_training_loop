import torch 
import numpy as np
from pathlib import Path
import math 
import pydub
import random
from torchvision import datasets


__all__=['build_dataloaders']

def load_audio_frames(filename, max_frames = 300, evalmode=False, num_eval=10):
    # https://github.com/clovaai/voxceleb_trainer

    max_audio = max_frames * 80
    audio  = np.array(pydub.AudioSegment.from_mp3(filename).get_array_of_samples())
    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
        audio       = np.pad(audio, (shortage, shortage), 'constant', constant_values=0)
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = np.stack(feats,axis=0)
    feat = torch.FloatTensor(feat).squeeze()
    return feat

def load_images_softmax(x):
    return load_audio_frames(x, max_frames=300) if Path(x).suffix in [".mp3", ".wav"] else np.load(x, allow_pickle = True)[None]

def build_dataloaders(train_dir, test_dir, batch_size):

    train_softmax_dataset = datasets.DatasetFolder(train_dir, loader = load_images_softmax, extensions=(".mp3",))
    train_softmax_loader = torch.utils.data.DataLoader(train_softmax_dataset, batch_size = batch_size, shuffle = True)

    test_softmax_dataset = datasets.DatasetFolder(test_dir, loader = load_images_softmax, extensions=(".mp3",))
    test_softmax_loader = torch.utils.data.DataLoader(test_softmax_dataset, batch_size = batch_size, shuffle = False)

    train_softmax_loader.idx_to_class = {i: c for c, i in train_softmax_dataset.class_to_idx.items()}
    test_softmax_loader.idx_to_class = {i: c for c, i in test_softmax_dataset.class_to_idx.items()}

    return train_softmax_loader, test_softmax_loader
