# create train and test dataloader
#     using ImageFolder
#     using DataLoader

import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

def create_compose_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(size = (64, 64)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor()
    ])

    return train_transform, test_transform


def create_dataloaders(train_dir: str,
                    test_dir: str,
                    train_transform: torchvision.transforms.Compose,
                    test_transform: torchvision.transforms.Compose,
                    batch_size: int,
                    num_workers: int):

    # create train data from image folder
    train_data = ImageFolder(
        root = train_dir,
        transform=train_transform,
        target_transform=None
    )

    # create test data from image folder
    test_data = ImageFolder(
        root = test_dir,
        transform = test_transform,
        target_transform = None
    )

    # get train data class names
    class_names = train_data.classes

    # create train dataloader
    train_dataloader = DataLoader(
        train_data,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = True
    )

    # create test dataloader
    test_dataloader = DataLoader(
        test_data,
        batch_size = batch_size,
        num_workers = num_workers
    )

    return train_dataloader, test_dataloader, class_names
