import torch 
import numpy as np
import random
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
from .audiospectrogram import get_fft_spectrum_from_array, get_fft_spectrum_from_file, load_wav, build_buckets

from .dataaugmentation import random_inversion, random_mask, normalize, channels_last, to_torch_tensor

def load_images_softmax(x, load_from_wav=False, gain_augmentation=False):
    # img = np.load(x, allow_pickle = True)
    if load_from_wav:
        try:
            buckets = build_buckets(10, 1, 0.01)
            img = get_fft_spectrum_from_file(x, buckets)
        except:
            return np.array([])
    else:
        try:
            img = np.load(x, allow_pickle = True)
        except:
            print(x)

    return np.expand_dims(img, 0)


class RandomSampler(Sampler):
  def __init__(self, data_source, seed=42):
    self.data_source = data_source
    self.seed = seed
    
    
  def set_seed(self):
    self.seed = 42

  def __iter__(self):
    n = len(self.data_source)
    self.indexes = list(range(n))
    random.Random(self.seed).shuffle(self.indexes)
    self.seed += 1
    return iter(self.indexes)

  def __len__(self):  
    return len(self.data_source)

def build_dataloaders(train_dir, test_dir, batch_size, data_augmentation=True, drop_last=False, *args, **kwargs):
    if data_augmentation:
        softmax_transforms_train = transforms.Compose([
                random_inversion, 
                random_mask, 
                #normalize,
                channels_last,
                transforms.ToTensor(),
                #to_torch_tensor
                ])

        softmax_transforms_test = transforms.Compose([
                #normalize,
                channels_last,
                transforms.ToTensor(),
                #to_torch_tensor
                ])
    else:
        softmax_transforms_train = transforms.Compose([
                normalize,
                transforms.ToTensor(),
                ])

        softmax_transforms_test = transforms.Compose([
                normalize,
                transforms.ToTensor(),
                ])
    try:
        seed = kwargs['seed']
    except:
        seed = random.randint(1, 2009)
    train_softmax_dataset = datasets.DatasetFolder(train_dir,
                                                   transform=softmax_transforms_train,
                                                   loader=load_images_softmax,
                                                   extensions=(".npy",))
    torch.manual_seed(seed)
    s = RandomSampler(train_softmax_dataset, seed)
    if drop_last == False:
        if len(train_softmax_dataset) % batch_size == 1:
            print('last batch will be sized 1, changing drop_last to true')
            drop_last = True 
    train_softmax_loader = torch.utils.data.DataLoader(train_softmax_dataset,
                                                       batch_size=batch_size,
                                                       drop_last=drop_last,
                                                       sampler=s)

    test_softmax_dataset = datasets.DatasetFolder(test_dir,
                                                  transform=softmax_transforms_test,
                                                  loader=load_images_softmax,
                                                  extensions=(".npy",))

    test_softmax_loader = torch.utils.data.DataLoader(test_softmax_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      drop_last=False)

    train_softmax_loader.idx_to_class = {i: c for c, i in train_softmax_dataset.class_to_idx.items()}
    test_softmax_loader.idx_to_class = {i: c for c, i in test_softmax_dataset.class_to_idx.items()}

    return train_softmax_loader, test_softmax_loader
