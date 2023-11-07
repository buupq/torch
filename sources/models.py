# some vision models

import torch
import torchvision
from torch import nn

class tinyVGG(nn.Module):

    """tinyVGG class
    Args:
        name: model name, default is tinyVGG
        inp_shape: input shape, which is the number of color channels
        out_shape: number of training class
        hidden_units: number of hidden units"""
    
    def __init__(self, inp_shape: int, out_shape: int, hidden_units=10, name: str="tinyVGG", INFO: bool=True):
        super().__init__()

        self.name = name

        if INFO:
            print(f"[INFO] creating {self.name} model...")

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16, out_features=out_shape)
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.block_2(self.block_1(x)))

    

# create functions to create pretrained efficinetNet model of various version
def create_effnet(effnet_version: int=0,
                 num_class_names: int=3,
                 device: torch.device="cuda",
                 model_name: str="",
                 INFO:bool=True):
    
    """create pretrained EfficientNet models
    Args:
        effnet_version (int): target EfficientNet version
        num_class_names: number of class name in new model
        device: torch.device
    Return:
        model: pretrained model with frozen model feature parameters
        model_transforms: data transformation method from pretrained model"""
    
    # get weights and model
    if effnet_version == 0:
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=weights)
    elif effnet_version == 1:
        weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
        model = torchvision.models.efficientnet_b1(weights=weights)
    elif effnet_version == 2:
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights)
    
    # get transformation from the weight
    model_transforms = weights.transforms()
    
    # freeze feature layer parameters
    for para in model.features.parameters():
        para.requires_grad = False
    
    # change classifier head
    if effnet_version == 0 or effnet_version == 1:
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=num_class_names, bias=True)
        ).to(device)
    elif effnet_version == 2:
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3, inplace=True),
            torch.nn.Linear(in_features=1408, out_features=num_class_names, bias=True)
        ).to(device)

    # assign model name if there is none
    if not model_name:
        if effnet_version == 0:
            model_name = "EfficientNet_B0"
        elif effnet_version == 1:
            model_name = "EfficientNet_B1"
        elif effnet_version == 2:
            model_name = "EfficientNet_B2"
            
    model.name = model_name
    
    # send model to device
    model = model.to(device)
    
    if INFO:
        print(f"[INFO] creating {model.name}...")
    
    return model, model_transforms