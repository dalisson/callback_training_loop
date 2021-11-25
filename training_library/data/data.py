from training_library.data import dataloaders
from .dataloaders.audiosoftmaximage import build_softmax_image_dataloader
from .dataloaders.audiosoftmax import build_audio_dataloaders
from .dataloaders.swave_dataloder import build_swave_dataloader
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

class Data(object):
    '''
    Data class for runner, wrapper around the train and valid dataloaders

    '''

    def __init__(self, dataloaders, bs, n_classes=None, data_type='img'):
        self.train_dl, self.valid_dl = dataloaders
        self.n_classes = n_classes
        self.data_type = data_type  
        self.bs = bs

    @classmethod
    def from_audio_images(cls, train_dir, test_dir, b_size, data_aug=True, drop_last=False, **kwargs):
        '''
        Builds Data class for audio images to be used with Softmax
        '''
        dataloaders = build_softmax_image_dataloader(train_dir=train_dir,
                                                     test_dir=test_dir,
                                                     batch_size=b_size,
                                                     data_augmentation=data_aug,
                                                     drop_last=drop_last,
                                                     **kwargs)
    

        classes = getattr(dataloaders[0], 'idx_to_class', None)
        if classes:
            n_classes = len(list(classes.keys()))
        else:
            n_classes = None
        return cls(dataloaders=dataloaders, bs=b_size, n_classes=n_classes, data_type='img')

    @classmethod
    def from_audio(cls, train_dir, test_dir, b_size):
        '''
        Builds Data class for audio images to be used with Softmax
        '''
        dataloaders = build_audio_dataloaders(train_dir=train_dir,
                                              test_dir=test_dir,
                                              batch_size=b_size)
        classes = getattr(dataloaders[0], 'idx_to_class', None)
        if classes:
            n_classes = len(list(classes.keys()))
        else:
            n_classes = None
        return cls(dataloaders=dataloaders, n_classes=n_classes, data_type='audio')
    
    @classmethod
    def for_audio_separation(cls, train_dir, eval_dir, sample_rate, batch_size):
        dataloaders = build_swave_dataloader(train_dir, eval_dir, sample_rate, batch_size)
        return cls(dataloaders, batch_size, None, "wav")


    def filter_classes_percentage(self, p=1, seed=42):
        #set np seed
        np.random.seed(seed)
        classes = np.array([c for c in self.train_dl.dataset.class_to_idx.keys()])
        trial = np.array(np.random.binomial(1, 1-p, len(classes)), dtype=bool)
        removed_classes = {c : self.train_dl.dataset.class_to_idx[c] for c in classes[trial]}

        remaining_classes = classes[(trial == False)]

        new_class_to_idx = {c: i for i, c in enumerate(remaining_classes)}

        self._filter_class_from_dataset(self.train_dl, removed_classes, new_class_to_idx)
        self._filter_class_from_dataset(self.valid_dl, removed_classes, new_class_to_idx)
        self.n_classes = len(self.train_dl.dataset.class_to_idx.keys())

    def _filter_class_from_dataset(self, dataloader, class_to_idx, new_class_to_idx):
        classes = class_to_idx.keys()
        filtered_samples = [x for x in dataloader.dataset.samples if x[0].split('/')[-2] not in classes]


        reversed_c_to_idx = {i : k for k, i in dataloader.dataset.class_to_idx.items()}
        label1 = [x[1] for x in filtered_samples]

        label_classes = [reversed_c_to_idx[l] for l in label1]
        labels = [new_class_to_idx[l] for l in label_classes]

        sample_location = [x[0] for x in filtered_samples]

        
        assert len(sample_location) == len(labels)
        #resetting the dataset
        dataloader.dataset.samples = [(x, y) for x, y in zip(sample_location, labels)]
        dataloader.dataset.targets = labels
        dataloader.dataset.class_to_idx = new_class_to_idx
        dataloader.class_to_idx = new_class_to_idx

    def show_batch(self, **kwargs):
        '''
        Show a batch of samples
        '''
        if self.data_type == 'img':
            self._show_image_batch(**kwargs)
        else:
            return 'Not Implemented for Data Type'

    def _show_image_batch(self, figsize=None):
        '''
        Show a batch of images
        '''
        figsize = (8, 8) if not figsize else figsize
        x, y = next(iter(self.train_dl))
        _, axes = plt.subplots(2, 2, figsize=figsize)
        for i, ax in enumerate(axes.flatten()):
            s_x, s_y = x[i], y[i]
            transposed = np.transpose(s_x.detach().cpu().numpy(), (1, 2, 0))
            n_channels = transposed.shape[-1]
            if n_channels == 1:
                transposed = np.squeeze(transposed, axis=-1)
            ax.imshow(transposed)
            ax.set_title(s_y.detach().cpu().numpy())
    
    def append_data(self, folder, f_type='.npy'):
        dataset = self.train_dl.dataset
        fo = Path(folder)
        new_files = []
        for f in fo.iterdir():
            for item in f.iterdir():
                if item.name.endswith(f_type):
                    new_files.append(str(item.resolve()))
        new_file_classes = [(lambda x: x.split('/')[-2])(x) for x in new_files]
        new_file_idx = [dataset.class_to_idx[x] for x in new_file_classes]
        assert len(new_file_idx) == len(new_files)

        new_samples = [(x, y) for x, y in zip(new_files, new_file_idx)]
        dataset.samples.extend(new_samples)
