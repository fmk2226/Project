from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

def make_folds(train_df,n_splits=5,seed=42):
    """ Yield (fold number, training indices, validation indices)
    
    Args:
        train_df: training dataframe
        n_spilits: number of folds
        seed: random seed
    """
    splitter=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    for fold,(train_idx,valid_idx) in enumerate(splitter.split(train_df['image'],train_df['label'])):
        yield fold,train_idx,valid_idx

def get_fold_indices(train_df,fold,n_splits=5,seed=42):
    """Return the training and validation row indices for one stratified fold.

    Args:
        train_df: DataFrame
        fold: Zero-based fold number to use as the validation fold
        n_splits: Total number of folds
        seed: Random seed used when shuffling samples before splitting

    Returns:
        A tuple (train_idx, valid_idx) of NumPy arrays. Each array contains
        zero-based row positions in train_df
    """
    if not 0<=fold<n_splits:
        raise ValueError(f"validation fold must be between 0 and {n_splits-1}")
    for current_fold,train_idx,valid_idx in make_folds(train_df,n_splits,seed):
        if current_fold==fold:
            return train_idx,valid_idx