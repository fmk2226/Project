import collections
from d2l import torch as d2l
import shutil
from pathlib import Path
import os
import math

def load_data(BASE_DIR,demo=True):
    d2l.DATA_HUB['dog_tiny']=(d2l.DATA_URL+'kaggle_dog_tiny.zip','0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')
    data_root=Path(BASE_DIR)/'data'
    data_root.mkdir(parents=True,exist_ok=True)
    data_dir=data_root/'kaggle_dog_tiny'
    if demo and not data_dir.exists():
        archive_path=Path(d2l.download('dog_tiny',folder=str(data_root)))
        shutil.unpack_archive(str(archive_path),str(data_root))
    elif demo==False:
        data_dir=data_root/'dog-breed-identification'
    return data_dir

def read_csv_labels(fname):
    """read fname to get a label dict"""
    with open(fname,'r') as f:
        #skip first row
        lines=f.readlines()[1:]
    tokens=[l.rstrip().split(',') for l in lines]
    return dict((name,label) for name,label in tokens)

def copyfile(filename,target_dir):
    os.makedirs(target_dir,exist_ok=True)
    target_path=os.path.join(target_dir,os.path.basename(filename))
    if not os.path.exists(target_path):
        shutil.copy(filename,target_dir)

def reorg_train_valid(data_dir,labels,valid_ratio):
    n=collections.Counter(labels.values()).most_common()[-1][1]
    n_valid_per_label=max(1,math.floor(n*valid_ratio))
    label_count={}
    for train_file in sorted(os.listdir(os.path.join(data_dir,'train'))):
        label=labels[train_file.split('.')[0]]
        fname=os.path.join(data_dir,'train',train_file)
        copyfile(fname,os.path.join(data_dir,'train_valid_test','train_valid',label))
        if label not in label_count or label_count[label]<n_valid_per_label:
            copyfile(fname,os.path.join(data_dir,'train_valid_test','valid',label))
            label_count[label]=label_count.get(label,0)+1
        else:
            copyfile(fname,os.path.join(data_dir,'train_valid_test','train',label))
    return n_valid_per_label

def reorg_test_valid(data_dir):
    for test_file in sorted(os.listdir(os.path.join(data_dir,'test'))):
        fname=os.path.join(data_dir,'test',test_file)
        copyfile(fname,os.path.join(data_dir,'train_valid_test','test','unknown'))

def reorg_dog_breed_data(data_dir,valid_ratio):
    labels=read_csv_labels(os.path.join(data_dir,'labels.csv'))
    print('train samples:',len(labels))
    print('classes:',len(set(labels.values())))
    reorg_train_valid(data_dir,labels,valid_ratio)
    reorg_test_valid(data_dir)