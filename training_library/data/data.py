from .dataloaders.audiosoftmaximage import build_softmax_image_dataloader

class Data(object):
    '''
    Data class for runner, wrapper around the train and valid dataloaders

    '''

    def __init__(self, dataloaders, n_classes=None):
        self.train_dl, self.valid_dl = dataloaders
        self.n_classes = n_classes

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
        return cls(dataloaders=dataloaders, n_classes=n_classes)
