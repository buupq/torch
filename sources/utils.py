import os
from pathlib import Path
import torch
import torchvision
from PIL import Image
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt

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


import os, pwd
from tensorboard import notebook
import getpass
from IPython.core.display import display, HTML
 
def get_pid_owner(pid):
    # the /proc/PID is owned by process creator
    proc_stat_file = os.stat("/proc/%d" % pid)
    # get UID via stat call
    uid = proc_stat_file.st_uid
    # look up the username from uid
    username = pwd.getpwuid(uid)[0]
    
    return username
 
def get_tb_port(username):
    
    for tb_nb in notebook.manager.get_all():
        if get_pid_owner(tb_nb.pid) == username:
            return tb_nb.port
    
def tb_address():
    
    username = getpass.getuser()
    tb_port = get_tb_port(username)
    
    address = "https://jupyter.olcf.ornl.gov" + os.environ['JUPYTERHUB_SERVICE_PREFIX'] + 'proxy/' + str(tb_port) + "/"
 
    address = address.strip()
    
    display(HTML('<a href="%s">%s</a>'%(address,address)))
    




def save_model(model:torch.nn.Module,
               model_dir: str,
               model_name_grid: []="",
               over_write: bool=True,
               INFO: bool=True):
    
    """save trained model
    Args:
        model: trained model
        model_dir: directory to save model (str)
        model_name_grid: grid list in model training grid
        over_write: overwriting the existing model
        INFO: extra info printout
    Returns:
        model_save_path (str): path to saved model"""

    model_dir = Path(model_dir)
    if model_dir.is_dir():
        if INFO:
            print(f"[INFO] {model_dir} exists.")
    else:
        if INFO:
            print(f"[INFO] creating folder to save model {model.name}...")
        model_dir.mkdir(parents=True, exist_ok=True)

    # path to saved model
    model_save_path = Path(model_dir) / ("_".join(model_name_grid) + ".pth")

    # check if the model exist and prompt overwritten
    write_model = True
    if model_save_path.is_file():
        if not over_write:
            write_model = False
            if INFO:
                print(f"[INFO] {model_save_path} exists. Ovewriting...")
            else:
                print(f"[INFO] {model_save_path} exists. Skipp writing.")

    # write model to file
    if write_model:
        if INFO:
            print(f"save model to: {model_save_path}...")
        torch.save(
            obj = model.state_dict(),
            f = model_save_path
        )

    return model_save_path






def load_saved_model(model: torch.nn.Module,
                     saved_model_path: str):

    model.load_state_dict(torch.load(saved_model_path))



# create writer
def create_writer(grid_names: []="",
                 INFO: bool=True):
    """create a tensorboard writer
    Args:
        grid_names: list of grid
    Returns:
        Tensorboard SummaryWriter"""
    
    log_dir = "writerLog"
    for grid_name in grid_names:
        log_dir = os.path.join(log_dir, grid_name)

    if INFO:
        print(f"[INFO] creating writer log at: {log_dir}")
    
    return SummaryWriter(
        log_dir=log_dir
    )


def predict_label(model: torch.nn.Module,
                  image_path: str,
                  class_names: list,
                  transforms: torchvision.transforms,
                  device: torch.device,
                  INFO: bool=False):

    """predict label from a model
    Args:
        model: trained model
        image_path: path to the image
        class_names: list of class name in the train dataset
        transforms: image transformation method
        device: device where model and image reside
        INFO: extra information printing flag including true label and image rendering
    Returns:
        label: predicted label"""
    
    # get correct label
    correct_label = image_path.parent.stem
    
    # switch to evaluation mode
    model.eval()
    with torch.inference_mode():
        # open image with PIL
        with Image.open(image_path) as img:
            # transform image to torch tensor
            img_transformed = transforms(img).unsqueeze(dim=0).to(device)
            # compute logit
            y_logit = model(img_transformed)
            # print logit for debugging
            if INFO:
                print(y_logit)
            # predict label index
            y_pred = torch.argmax(torch.softmax(y_logit, dim=1))
            # get label
            label = class_names[y_pred]

            # plot the image with correct and predicted labels
            if INFO:
                plt.imshow(img_transformed.squeeze().permute(1,2,0).to("cpu").numpy())
                plt.title(f"correct label: {correct_label} | predicted label: {label} | {correct_label == label}")

    return label



def plot_loss_acc(results: dict):

    """plot train and test loss and accuracy from result dictionary
    Args:
        results: dictionary of train and test loss and accuracy"""

    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    
    ax[0].plot(results["train_loss"], '-^', label="train_loss")
    ax[0].plot(results["test_loss"], '-*', label="test_loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    
    ax[1].plot(results["train_acc"], '-^', label="train_acc")
    ax[1].plot(results["test_acc"], '-*', label="test_acc")
    ax[0].set_xlabel("epoch")
    ax[1].legend()

