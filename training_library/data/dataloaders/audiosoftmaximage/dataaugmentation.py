import torch
import numpy as np 
import torchaudio
from .audiospectrogram import get_fft_spectrum_from_array, load_wav,\
                              get_fft_spectrum_from_file, build_buckets

def random_inversion(img, prob = 0.5):
    if not img.shape[0] == 0:
        if np.random.rand() <= prob:
            return img[:, :, ::-1]
        else:
            return img
    else:
        return img
def random_mask(img, max_freq_masks = 2, max_time_masks = 2, min_mask_size = 5, max_mask_size = 40):
    if not img.shape[0] == 0:
        n_freq_masks = np.random.choice([i for i in range(max_freq_masks+1)], size = 1).item()
        n_time_masks = np.random.choice([i for i in range(max_time_masks+1)], size = 1).item()
        for _ in range(n_freq_masks):
            mask_freq_size = np.random.randint(low = min_mask_size, high = max_mask_size)
            mask_freq_loc = np.random.randint(low = 0, high = img.shape[1]-mask_freq_size-1)
            img[:, mask_freq_loc:mask_freq_loc+mask_freq_size, :] = np.zeros((1, mask_freq_size, img.shape[2]))
        
        for _ in range(n_time_masks):
            mask_time_size = np.random.randint(low = min_mask_size, high = max_mask_size)
            mask_time_loc = np.random.randint(low = 0, high = img.shape[2]-mask_time_size-1)
            img[:, :, mask_time_loc:mask_time_loc+mask_time_size] = np.zeros((1, img.shape[1], mask_time_size))
        return img
    else:
        return img

def apply_gain(filename, sr = 8000):
    gain_db = np.random.uniform(low = -15, high = 15)
    audio = load_wav(filename, sr)
    start = np.random.randint(low = 0, high = len(audio)-1)
    end = np.random.randint(low = start, high = len(audio))
    a0 = audio[:start]  
    g = audio[start:end]
    a1 = audio[end:]
    g = torchaudio.functional.gain(g, gain_db = gain_db)
    gain = np.concatenate((a0, g, a1))
    return gain

def normalize(img):
    if not img.shape[0] == 0:
        img =  ((img - img.min()) / (img.max() - img.min())).transpose(1, 2, 0)
        return img
    else:
        print("Audio with error.")
        #return np.zeros([512, 300])

def channels_last(img):
    transp = img.transpose(1, 2, 0)
    return np.ascontiguousarray(transp)

def to_torch_tensor(img):
    return torch.tensor(img)
