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
