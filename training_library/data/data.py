from .dataloaders.audiosoftmaximage import build_softmax_image_dataloader

class Data(object):
    '''
    Data class for runner, wrapper around the train and valid dataloaders

    '''

    def __init__(self, dataloaders, n_classes=None):
        self.train_dl, self.valid_dl = dataloaders
        self.n_classes = n_classes

    def get_n_classes(self):
        assert self.n_classes is not None
        return self.n_classes

    @classmethod
    def build_softmax_data_audioimage(cls, train_dir, test_dir, b_size, data_aug=True):
        dataloaders = build_softmax_image_dataloader(train_dir=train_dir,
                                                     test_dir=test_dir,
                                                     batch_size=b_size,
                                                     data_augmentation=data_aug)
        classes = getattr(dataloaders[0], 'classes', None)
        if classes:
            n_classes = len(classes)
        return cls(dataloaders=dataloaders, n_classes=n_classes)
