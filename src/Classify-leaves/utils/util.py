def inf_loop(data_loader):
    """ Wrapper function for a endless data_loader

    Args:
        data_loader: data iterator

    Yield:
        A batch from data_loader, typically an (images,labels) tuple
    """
    while(True):
        yield from data_loader

class MetricTracker:
    def __init__(self,*keys):
        pass

    def reset(self):
        pass

    def update(self,key,value,n=1):
        pass

    def avg(self,key):
        pass

    def result(self):
        pass