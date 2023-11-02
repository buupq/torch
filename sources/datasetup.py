# download data
from pathlib import Path
from zipfile import ZipFile
import requests
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

### download data from url ###
def download_data(source: str,
                 destination: str="images",
                 remove_source: bool=True,
                 INFO: bool=True):
    
    """Download file from an url to data, extract files to images folder
    Args:
        source: url to the data source
    Returns:
        image_path: folder to store train and test images"""

    # setup data and image paths
    data_path = Path("data")
    image_path = data_path / destination

    # check if image_path exist and create the dir
    if image_path.is_dir() and len(os.listdir(image_path)) > 0 :
        if INFO:
            print(f"[INFO] {image_path} exist.")
    else:
        if INFO:
            print(f"[INFO] creating {image_path}...")
        image_path.mkdir(parents=True, exist_ok=True)

        zip_file_name = Path(source).name
        zip_file_path = data_path / zip_file_name
        if INFO:
            print(f"[INFO] downloading {zip_file_name} to {data_path}...")
        with open(zip_file_path, "wb") as f:
            request = requests.get(url=source)
            f.write(request.content)
        if INFO:
            print(f"[INFO] extracting {zip_file_name} to {image_path}...")
        with ZipFile(zip_file_path) as zip_ref:
            zip_ref.extractall(image_path)
    
        if remove_source:
            if INFO:
                print(f"[INFO] removing {zip_file_name}...")
            os.remove(zip_file_path)

    return image_path


### create dataloaders ###
def create_dataloaders(train_dir: str,
                    test_dir: str,
                    train_transforms: torchvision.transforms.Compose,
                    test_transforms: torchvision.transforms.Compose,
                    batch_size: int=32,
                    num_workers: int=1,
                    INFO: bool=True):

    """create train and test dataloader from train and test directory using ImageFolder
    Args:
        train_dir: train directory
        test_dir: test directory
        train_transforms: composed transforms for train dataset
        test_transforms: composed transforms for test dataset
        batch_size: batch size, default = 32
        num_workers: number of processor, default=1
    Returns:
        train_dataloader: dataloader of train dataset
        test_dataloader: dataloader of test dataset
        class_names: class names in the train dataset"""

    
    # create train data from image folder
    train_data = ImageFolder(
        root = train_dir,
        transform=train_transforms,
        target_transform=None
    )

    # create test data from image folder
    test_data = ImageFolder(
        root = test_dir,
        transform = test_transforms,
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
    
    if INFO:
        print(
            f"[INFO] creating dataloaders... \n"
            f"train_dataloader: {train_dataloader} \n"
            f"test_dataloader: {test_dataloader} \n"
            f"number of class_names: {len(class_names)}"
        )

    return train_dataloader, test_dataloader, class_names
