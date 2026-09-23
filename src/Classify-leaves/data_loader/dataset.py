import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class Leafdataset(Dataset):
    """ Load leaf images listed in a CSV file

    Args:
        csv_file: Path to train.csv or test.csv.
        data_dir: Dataset root directory. CSV & image paths are relative to it.
        label_to_id: Mapping from class names to integer IDs. Required for
            labeled training data; not used for the unlabeled test set.
        transform: Optional function applied to each image after it is loaded.

    Returns from __getitem__:
        For training data, (transformed image, integer class ID).
        For test data, transformed image only.
    """
    def __init__(self,csv_file,data_dir,label_to_id=None,transform=None):
        self.df=pd.read_csv(csv_file)
        self.data_dir=data_dir
        self.label_to_id=label_to_id
        self.transform=transform

    def __len__(self):
        """ Count number of samples in the dataset
        
        Returns:
            A Int of sample numbers
        """
        return len(self.df)

    def __getitem__(self, index):
        """ Load one image and its label, if available.
        
        Args:
            index: Zero-based row index in the CSV file.

        Returns:
            For a labeled CSV, a tuple of (transformed image, integer class ID).
            For an unlabeled CSV, the transformed image only.
        """
        image_path=os.path.join(self.data_dir,self.df.iloc[index,0])
        img=Image.open(image_path).convert("RGB")
        if self.transform!=None:
            img=self.transform(img)
        if "label" not in self.df.columns:
            return img
        label=self.df.iloc[index,1]
        if self.label_to_id==None:
            raise ValueError("Training set requires label_to_id mapping")
        return img,self.label_to_id[label]