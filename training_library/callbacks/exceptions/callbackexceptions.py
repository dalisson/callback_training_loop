class DeviceException(Exception):
    '''
    Exception for control flow of training
    '''
    def __init__(self):
        super().__init__('Device not set, check cuda callback')