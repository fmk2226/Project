import pandas as pd

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
        self._data=pd.DataFrame(index=keys,columns=['total','counts','average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:]=0

    def update(self,key,value,n=1):
        self._data['total'][key]+=value*n
        self._data['counts'][key]+=n
        self._data['average'][key]=self._data['total']/self._data['counts']

    def avg(self,key):
        return self._data['average']

    def result(self):
        return dict(self._data['average'])