
import torch
import torch.nn as nn
from torch.optim import Adam


def create_training_components(
    model,
    learning_rate=0.001,
    weight_decay=0.0001
):
    """
    Create loss function and optimizer for MLP training.
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    return criterion, optimizer


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    """
    Train the model for one epoch.
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy
