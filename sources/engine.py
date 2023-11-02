#################
## main engine ##
#################

import torch
from torch.utils.tensorboard.writer import SummaryWriter as writer
from torch import nn
from tqdm.auto import tqdm


### train step function ###
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):

    """training step
    Args:
        model: input model
        dataloader: train dataloader
        loss_fn: loss function
        optimizer: optimizer
        device: training device
    Returns:
        train_loss: training loss
        train_acc: training accuracy"""
    
    # switch to train mode
    model.train()

    # initialize train loss and accuracy
    train_loss, train_acc = 0, 0

    # loop over dataloader batch
    for batch, (X, y) in enumerate(dataloader):

        # send data to device
        X, y = X.to(device), y.to(device)

        # compute train logit
        train_logit = model(X)

        # compute loss
        loss = loss_fn(train_logit, y)

        # accumulate train loss
        train_loss += loss.item()

        # zero optimizer gradient
        optimizer.zero_grad()

        # back propagation
        loss.backward()

        # optimizer steps
        optimizer.step()

        # predict image labels
        y_pred = torch.argmax(torch.softmax(train_logit, dim=1), dim=1)

        # compute and accumulate accuracy based on predicted labels
        train_acc += (y_pred == y).sum().item() / len(y_pred)

    # adjust train loss and accuracy
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    # return train loss and accuracy
    return train_loss, train_acc

### test step function ###
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
             device: torch.device):

    """testing step
    Args:
        model: input model
        dataloader: test dataloader
        loss_fn: loss function
        device: testing device 
    Returns:
        test_loss: testing loss
        test_acc: testing accuracy"""

    # switch to evaluation mode
    model.eval()

    # initialize test loss and accuracy
    test_loss, test_acc = 0, 0

    # loop over dataloader batch
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):

            # send data to device
            X, y = X.to(device), y.to(device)

            # compute test logit
            test_logit = model(X)

            # compute and accumulate test loss
            test_loss += loss_fn(test_logit, y).item()

            # compute test labels
            y_pred = torch.argmax(torch.softmax(test_logit, dim=1), dim=1)

            # compute and accumulate test accuracy
            test_acc += (y_pred == y).sum().item() / len(y_pred)

        # adjust test loss and accuracy
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    # return test loss and accuracy
    return test_loss, test_acc
    

### training function ###
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          INFO:bool=True):
    
    """training function
    Args:
        model: input model
        train_dataloader: train dataloader
        test_dataloader: test dataloader
        loss_fn: loss function
        optimizer: optimizer
        epochs: number of epochs
        device: training device
    Returns:
        results: dictionary of train, test loss and accuracy"""

    
    # initialize result dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # loop over epochs
    for epoch in tqdm(range(epochs)):

        # get train loss and acc
        train_loss, train_acc = train_step(
            model = model,
            dataloader = train_dataloader,
            loss_fn = loss_fn,
            optimizer = optimizer,
            device = device
        )

        # get test loss and acc
        test_loss, test_acc = test_step(
            model = model,
            dataloader = test_dataloader,
            loss_fn = loss_fn,
            device=device
        )

        # print out result in each epoch
        if INFO:
            print(
                f"epoch: {epoch} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

        # update result dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # return result dictionary
    return results


### training function with tensorboard writer ###
def train_tsb_writer(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter,
          INFO: bool=True):
    
    """training function including tensorboard writer
    Args:
        model: input model
        train_dataloader: train dataloader
        test_dataloader: test dataloader
        loss_fn: loss function
        optimizer: optimizer
        epochs: number of epochs
        device: training device
    Returns:
        results: dictionary of train, test loss and accuracy"""

    
    # initialize result dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # loop over epochs
    for epoch in tqdm(range(epochs)):

        # get train loss and acc
        train_loss, train_acc = train_step(
            model = model,
            dataloader = train_dataloader,
            loss_fn = loss_fn,
            optimizer = optimizer,
            device = device
        )

        # get test loss and acc
        test_loss, test_acc = test_step(
            model = model,
            dataloader = test_dataloader,
            loss_fn = loss_fn,
            device=device
        )

        # print out result in each epoch
        if INFO:
            print(
                f"epoch: {epoch} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

        # update result dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # add train and test Loss to writer
        writer.add_scalars(
            main_tag = "Loss",
            tag_scalar_dict = {
                "train_loss": train_loss,
                "test_loss": test_loss
            },
            global_step = epoch
        )

        # add train and test Acc to writer
        writer.add_scalars(
            main_tag = "Acc",
            tag_scalar_dict = {
                "train_acc": train_acc,
                "test_acc": test_acc
            },
            global_step=epoch
        )

    # close writer
    writer.close()
    
    # return result dictionary
    return results


