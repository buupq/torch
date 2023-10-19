import os
from pathlib import Path
import torch

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def list_file_tree(
    targ_dir=".",
    print_all_files=False,
    num_file_cap=2,
    print_dir_only=True):

    """list directories and files of a target directory
    Args:
        targ_dir: target directory
        print_all_files: print all available directory and files in targ_dir
        print_dir_only: print directory tree only
        num_file_cap: maximum number of files listed
    Examples:
        list_file_tree(targ_dir=".", print_dir_only=True)
        would return only list of directory
    """
    if not Path(targ_dir).is_dir():
        return f"{targ_dir}: does not exist!"

    for root, dirs, files in list(os.walk(targ_dir)):
        # print directories
        level = root.replace(str(targ_dir), "").count(os.sep)
        indent = " " * 3 * level
        print(f"{indent} {bcolors.OKBLUE} {root} {bcolors.ENDC}")

        if not print_dir_only:
        # print files in directory
            subindent = " " * 3 * (level + 1)
            num_files = 0
            if print_all_files:
                num_file_cap = len(files)
            for file in files:
                if num_files < num_file_cap:
                    print(f"{subindent} {bcolors.OKGREEN} {file} {bcolors.ENDC}")
                    num_files += 1
                    if num_files == num_file_cap:
                        print(f"{subindent} and {len(files)} other files...")


# download data
from pathlib import Path
from zipfile import ZipFile
import requests

def download_data(url):
    """Download file from an url to data, extract files to images folder
    Args:
        url: url to the data source
    Returns:
        image_path, train_path, test_path = download_data(url=url)
            image_path: folder to store train and test images
            train_path: folder to store train data
            test_path: folder to store test data
    """

    # create data directory
    data_path = Path("data")
    image_path = data_path / "images"
    image_path.mkdir(parents=True, exist_ok=True)

    # download zip file to data/
    request = requests.get(url)
    with open(data_path / "images.zip", "wb") as f:
        f.write(request.content)

    # extract images to data/images
    with ZipFile(data_path / "images.zip") as zip_ref:
        zip_ref.extractall(image_path)

    # train and test directories
    train_dir = image_path / "train"
    test_dir = image_path / "test"


    return image_path, train_dir, test_dir


def save_model(model:torch.nn.Module,
               model_dir: str):

    model_save_path = Path(model_dir) / (model.name + ".pth")

    torch.save(
        obj = model.state_dict(),
        f = model_save_path
    )

    return model_save_path

def load_saved_model(loaded_model: torch.nn.Module,
                     model_saved_path: str):

    loaded_model.load_state_dict(torch.load(model_saved_path))
