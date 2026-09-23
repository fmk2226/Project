from torchvision import transforms

#Normalization for pretrained ImageNet models
normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

def transform_train():
    """ Data augmentation for train set
    
    Returns:
        train transform method
    """
    transform=transforms.Compose([
        transforms.Resize((320,320)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180),  #leaves can occur at any position
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2,contrast=0.2)],p=0.5),
        transforms.RandomApply([transforms.RandomAffine(degrees=0,translate=(0.25,0.25),scale=(0.9,1.1))],p=0.5),
        transforms.ToTensor(),
        normalize
    ])
    return transform

def transform_test():
    """ Data augmentation for test set

    Returns:
        test transfrom method
    """
    transform=transforms.Compose([
        transforms.Resize((320,320)),
        transforms.ToTensor(),
        normalize
    ])
    return transform

def transform_valid():
    """ Data augmentation for valid set, the method is same as test
    
    Returns:
        valid transform method
    """
    transform=transforms.Compose([
            transforms.Resize((320,320)),
            transforms.ToTensor(),
            normalize
        ])
    return transform