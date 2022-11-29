import os
import pickle
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data_processing.dataset import ProjectDataSet
from models.resnet import *

from utils.utils import progress_bar

MODEL_NAME = 'best_model.pt'


def create_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch NNDL image classification challenge')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--early_stopping_patience', default=10, type=int,
                        help='Early stopping patience')
    parser.add_argument('--training_data_path', required=True,
                        help='input_folder containing the images')
    parser.add_argument('--training_label_path', required=True, help='the path to training label')
    parser.add_argument('--test_data_path', required=True,
                        help='input_folder containing the images')
    parser.add_argument('--checkpoint_path', required=True, help='checkpoint_path for the model')
    return parser


def checkpoint(
        net,
        history,
        checkpoint_path
):
    """Saves the current encoder and decoder models, along with idx_dict, which
    contains the char_to_index and index_to_char mappings, and the start_token
    and end_token values.
    """

    create_dir_if_not_exists(checkpoint_path)

    with open(os.path.join(checkpoint_path, MODEL_NAME), 'wb') as f:
        torch.save(net, f)

    with open(os.path.join(checkpoint_path, 'history.pickle'), 'wb') as f:
        pickle.dump(history, f)


def create_dir_if_not_exists(directory):
    """Creates a directory if it doesn't already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


# Training
def train(
        net,
        train_loader,
        criterion,
        optimizer,
        device
):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    train_average_loss = train_loss / total
    return train_average_loss, acc


def validate(
        net,
        val_loader,
        criterion,
        device
):
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    val_average_loss = val_loss / total
    return val_average_loss, acc


def predict(
        net,
        data_loader,
        device
):
    net.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            outputs = net(inputs.to(device))
            predictions.append(torch.argmax(outputs, dim=-1).detach().cpu().numpy())
    return pd.DataFrame(np.hstack(predictions), columns=['predictions'])


def main(args):
    # Data
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4707, 0.4431, 0.3708), (0.1577, 0.1587, 0.1783)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4707, 0.4431, 0.3708), (0.1577, 0.1587, 0.1783)),
    ])

    training_dataset = ProjectDataSet(
        image_folder_path=args.training_data_path,
        data_label_path=args.training_label_path,
        transform=transform_train,
        is_superclass=True
    )

    train_total = len(training_dataset)
    train_size = int(train_total * 0.9)
    val_size = train_total - train_size
    train_set, val_set = torch.utils.data.random_split(
        training_dataset, [train_size, val_size]
    )

    test_set = ProjectDataSet(
        image_folder_path=args.test_data_path,
        transform=transform_test,
        is_superclass=True
    )

    train_dataloader = DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=4
    )

    val_dataloader = DataLoader(
        val_set, batch_size=128, shuffle=True, num_workers=4
    )

    test_dataloader = DataLoader(
        test_set, batch_size=128, num_workers=4
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet50()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    training_loop(
        net,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        scheduler,
        args.epochs,
        device,
        args.early_stopping_patience,
        args.checkpoint_path
    )

    net = torch.load(os.path.join(args.checkpoint_path, MODEL_NAME))
    predictions_pd = predict(net, test_dataloader, device)
    predictions_pd.to_csv(
        os.path.join(args.checkpoint_path, 'predictions.csv'),
        index=False
    )


def update_metrics(
        history,
        train_loss,
        val_loss,
        train_acc,
        val_acc
):
    if 'train_loss' not in history:
        history['train_loss'] = []

    if 'val_loss' not in history:
        history['val_loss'] = []

    if 'train_acc' not in history:
        history['train_acc'] = []

    if 'val_acc' not in history:
        history['val_acc'] = []

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)


def training_loop(
        net,
        train_dataloader,
        test_dataloader,
        criterion,
        optimizer,
        scheduler,
        epochs,
        device,
        early_stopping_patience,
        checkpoint_path
):
    early_stopping_counter = 0
    best_val_loss = 1e6

    history = {}

    for epoch in range(0, epochs):

        train_loss, train_acc = train(net, train_dataloader, criterion, optimizer, device)
        val_loss, val_acc = validate(net, test_dataloader, criterion, device)
        scheduler.step()

        update_metrics(
            history,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc
        )

        if val_loss < best_val_loss:
            checkpoint(net, history, checkpoint_path)
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Stop the training if the val does not improve
        if early_stopping_counter > early_stopping_patience:
            print("Validation loss has not improved in {} epochs, stopping early".format(
                early_stopping_patience))
            print("Obtained lowest validation loss of: {}".format(best_val_loss))
            break

    return history


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
