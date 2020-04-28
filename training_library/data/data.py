from .dataloaders.audiosoftmaximage import build_softmax_image_dataloader
from matplotlib import pyplot as plt
import numpy as np

class Data(object):
    '''
    Data class for runner, wrapper around the train and valid dataloaders

    '''

    def __init__(self, dataloaders, n_classes=None, data_type='img'):
        self.train_dl, self.valid_dl = dataloaders
        self.n_classes = n_classes
        self.data_type = data_type

    @classmethod
    def from_audio_images(cls, train_dir, test_dir, b_size, data_aug=True):
        '''
        Builds Data class for audio images to be used with Softmax
        '''
        dataloaders = build_softmax_image_dataloader(train_dir=train_dir,
                                                     test_dir=test_dir,
                                                     batch_size=b_size,
                                                     data_augmentation=data_aug)
        classes = getattr(dataloaders[0], 'idx_to_class', None)
        if classes:
            n_classes = len(list(classes.keys()))
        else:
            n_classes = None
        return cls(dataloaders=dataloaders, n_classes=n_classes, data_type='img')

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
        _, axes = plt.subplots(8, 2, figsize=figsize)
        for i, ax in enumerate(axes.flatten()):
            s_x, s_y = x[i], y[i]
            transposed = np.transpose(s_x.detach().cpu().numpy(), (1, 2, 0))
            n_channels = transposed.shape[-1]
            if n_channels == 1:
                transposed = np.squeeze(transposed, axis=-1)
            ax.imshow(transposed)
            ax.set_title(s_y.detach().cpu().numpy())