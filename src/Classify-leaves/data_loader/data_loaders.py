import pandas as pd
from torch.utils.data import DataLoader
from .dataset import Leafdataset
from torch.utils.data import Subset
from .transforms import transform_train,transform_test,transform_valid
from .split import get_fold_indices

class LeafDataLoader(DataLoader):
    """ Generate training and validation Dataloader for a CSV file

    Args:
        csv_file: Path to train.csv or test.csv
        data_dir: Dataset root directory. CSV & image paths are relative to it
        batch_size: number of images in a single batch
        fold: Zero-based fold number to use as the validation fold
        n_splits: Total number of folds
        seed: Random seed
        shuffle: Whether to shuffle when loading the data or not
        num_workers: Number of worker threads

    Attributes:
        label_to_id: Mapping from class name to integer IDs
        Training iterator
        Valid iterator
    """
    def __init__(self,csv_file,data_dir,batch_size,shuffle=True,fold=0,n_splits=5,num_workers=8,Training=True,seed=42):
        df=pd.read_csv(csv_file)
        if Training:
            #creating label map
            classes=sorted(df['label'].unique())
            self.label_to_id={name:i for i,name in enumerate(classes)}

            #separating training set index and validation set index
            train_idx,valid_idx=get_fold_indices(df,fold=fold,n_splits=n_splits,seed=seed)

            train_ds=Leafdataset(csv_file,data_dir,self.label_to_id,transform_train())
            valid_ds=Leafdataset(csv_file,data_dir,self.label_to_id,transform_valid())
            train_ds=Subset(train_ds,train_idx)
            valid_ds=Subset(valid_ds,valid_idx)

            super().__init__(train_ds,batch_size,shuffle,num_workers=num_workers)
            self.valid_iter=DataLoader(valid_ds,batch_size,shuffle=False,num_workers=num_workers)

        #if testing    
        else:
            test_ds=Leafdataset(csv_file,data_dir,label_to_id=None,transform=transform_test())
            super().__init__(test_ds,batch_size,shuffle=False,num_workers=num_workers)

    def split_validation(self):
        """ Get the validation iterator

        Return:
            Validation iterator
        """
        return self.valid_iter