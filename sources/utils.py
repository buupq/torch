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
               model_dir: str):
    
    """save trained model
    Args:
        model: trained model
        model_dir: directory to save model (str)
    Returns:
        model_save_path (str): path to saved model"""

    model_dir = Path(model_dir)
    if model_dir.is_dir():
        print(f"[INFO] {model_dir} exists.")
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
        
    model_save_path = Path(model_dir) / (model.name + ".pth")

    torch.save(
        obj = model.state_dict(),
        f = model_save_path
    )

    return model_save_path



def load_saved_model(loaded_model: torch.nn.Module,
                     model_saved_path: str):

    loaded_model.load_state_dict(torch.load(model_saved_path))

    