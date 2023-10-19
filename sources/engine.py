
# main engine
import torch
from torch import nn

class tinyVGG(nn.Module):
    def __init__(self, name: str, inp_shape: int, out_shape: int, hidden_units=10):
        super().__init__()

        self.name = name

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

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):

    # switch to train mode
    model.train()

    # initialize train loss and accuracy
    train_loss, train_acc = 0, 0

    # loop over dataloader batch
    for batch, (X, y) in enumerate(dataloader):

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

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):

    # switch to evaluation mode
    model.eval()

    # initialize test loss and accuracy
    test_loss, test_acc = 0, 0

    # loop over dataloader batch
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):

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

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int=3):


    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    for epoch in range(epochs):

        train_loss, train_acc = train_step(
            model = model,
            dataloader = train_dataloader,
            loss_fn = loss_fn,
            optimizer = optimizer
        )

        test_loss, test_acc = test_step(
            model = model,
            dataloader = test_dataloader,
            loss_fn = loss_fn
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


        print(f"epoch: {epoch} | train loss: {train_loss:.3f} | train acc: {train_acc:.3f} | test loss: {test_loss:.3f} | test acc: {test_acc:.3f}")

    return results

