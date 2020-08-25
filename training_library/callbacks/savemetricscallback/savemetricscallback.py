from ..callback import Callback

class SaveMetricsCallback(Callback):
    order = 10
    def __init__(self, f):
        super(SaveMetricsCallback, self).__init__()
        self.f = f

    def begin_fit(self):
        '''
        sets up the file for saving
        '''
        header = ''
        for stage in self.stages:
            for metric in self.metrics[stage].keys():
                header += '{}_{},'.format(stage, metric)
        header = header[:-1]
        with open(self.f, 'w') as save_file:
            save_file.write(header)


    def after_epoch(self):
        '''
        Save everything at the end of the epoch
        '''
        for stage in self.stages:
            for metric in self.metrics[stage].keys():
                result = self.run.metrics[stage][metric]
                header += '{},'.format(result)
        header = header[:-1]
        with open(self.f, 'a') as save_file:
            save_file.write(header)
        