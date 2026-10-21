import pandas as pd
import random
import numpy as np
import torch

# TODO: add set_seed(seed) that seeds random, numpy and torch (incl. cuda) for reproducible K-fold runs
def set_seed(seed):
    """ Seed everythin random, numpy and torch for reproducible runs
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    """ Track averages of named metrics
    
    Args:
        *keys: Unique metric names

    Attributes:
        _data: DataFrame storing weighted total, accumulated count, and average for each metric
    """
    def __init__(self,*keys):
        self._data=pd.DataFrame(index=keys,columns=['total','counts','average'])
        self.reset()

    def reset(self):
        """ Reset all metric totals, counts, and averages to 0 """
        for col in self._data.columns:
            self._data[col].values[:]=0

    def update(self,key,value,n=1):
        """ Update a metric's weighted total, counts, and average.
        
        Args:
            key: Name of the metric.
            value: Scalar metric value, usually a batch's average.
            n: Usually the batch size.
        """
        self._data.loc[key,'total']+=value*n
        self._data.loc[key,'counts']+=n
        self._data.loc[key,'average']=self._data.loc[key,'total']/self._data.loc[key,'counts']

    def avg(self,key):
        """ Return the average of a metric
        
        Args:
            key: Name of the metric

        Returns:
            The average of a metric
        """
        return self._data.loc[key,'average']

    def result(self):
        """ Return the current averages of all metrics.
        
        Returns:
            A dict mapping metric names to their averages.
        """
        return dict(self._data['average'])