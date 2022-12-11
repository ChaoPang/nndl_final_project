import sys
import os
import pickle
import math
import argparse
import pandas as pd
import torch.nn.functional

from torch import nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from data_processing.recover_dataset import RecoverPredictionDataset
from models.finetune_pretrained import ReconstructClassificationModel

from utils.utils import progress_bar

import matplotlib.pyplot as plt

MODEL_NAME = 'best_model.pt'

sys.setrecursionlimit(10000)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch NNDL image classification challenge')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='Dropout rate')
    parser.add_argument('--freeze_weight', action='store_true',
                        help='Whether or not we freeze the weights of the pretrained model')
    parser.add_argument('--deep_feature', action='store_true',
                        help='Whether or not extract the deep feature')
    parser.add_argument('--img_size', default=8, type=int, help='Image Size')
    parser.add_argument('--img_upsampled_size', default=8, type=int,
                        help='img upsampled size by auto encoder')
    parser.add_argument('--data_path', required=True,
                        help='input_folder containing the CIFAR images')
    parser.add_argument('--checkpoint_path', required=True, help='checkpoint_path for the model')

    return parser


def checkpoint(
        net,
        history,
        checkpoint_path,
        model_name
):
    """Saves the current encoder and decoder models, along with idx_dict, which
    contains the char_to_index and index_to_char mappings, and the start_token
    and end_token values.
    """

    create_dir_if_not_exists(checkpoint_path)

    with open(os.path.join(checkpoint_path, model_name), 'wb') as f:
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
        cls_criterion,
        reconstruct_criterion,
        optimizer,
        device
):
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets, high_resolution_targets) in enumerate(train_loader):
        inputs, targets, high_resolution_targets = inputs.to(device), targets.to(
            device), high_resolution_targets.to(high_resolution_targets)

        optimizer.zero_grad()
        outputs, reconstruction_outputs = net(inputs)

        cls_loss = cls_criterion(
            outputs,
            targets
        )
        reconstruction_loss = reconstruct_criterion(
            reconstruction_outputs,
            high_resolution_targets
        )
        loss = cls_loss + reconstruction_loss
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
        cls_criterion,
        reconstruction_criterion,
        device
):
    net.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, high_resolution_targets) in enumerate(val_loader):
            inputs, targets, high_resolution_targets = inputs.to(device), targets.to(
                device), high_resolution_targets.to(device)

            outputs, reconstruction_outputs = net(inputs)
            cls_loss = cls_criterion(outputs, targets)
            reconstruction_loss = reconstruction_criterion(
                reconstruction_outputs,
                high_resolution_targets
            )

            val_loss += cls_loss.item() + reconstruction_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    val_average_loss = val_loss / total
    return val_average_loss, acc


def train_model(
        net,
        train_set,
        val_set,
        args,
        device
):
    train_dataloader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    val_dataloader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    cls_criterion = nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss()

    optimizer = optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=1e-4, eps=0.1
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    history = training_loop(
        net,
        train_dataloader,
        val_dataloader,
        cls_criterion,
        reconstruction_criterion,
        optimizer,
        scheduler,
        args.epochs,
        device,
        args.checkpoint_path
    )

    return history


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    train_set, val_set = create_datasets(args)

    # Initialize the model
    device = get_device()

    upscale_factor = int(args.img_upsampled_size // args.img_size)
    print(f'Upscale factor is {upscale_factor}\n')

    net = ReconstructClassificationModel(
        num_classes=257,
        deep_feature=args.deep_feature,
        upscale_factor=upscale_factor
    )
    net = net.to(device)

    history = train_model(net, train_set, val_set, args, device)

    plot_training_loss(history, args.checkpoint_path)


def plot_training_loss(history, checkpoint_path):
    # Plot training curve
    plt.figure()
    plt.plot(history['train_loss'], "ro-", label="Train")
    plt.plot(history['val_loss'], "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig(checkpoint_path + "/training_curve.png")


def create_datasets(
        args
):
    train_set = RecoverPredictionDataset(
        img_input_size=args.img_size,
        img_output_size=args.img_upsampled_size,
        data_folder=args.data_path
    )

    train_total = len(train_set)
    train_size = int(train_total * 0.8)
    val_size = train_total - train_size
    train_set, val_set = torch.utils.data.random_split(
        train_set, [train_size, val_size]
    )

    print(f'train_set size: {len(train_set)}')
    print(f'val_set size: {len(val_set)}')

    return train_set, val_set


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
        val_dataloader,
        cls_criterion,
        reconstruction_criterion,
        optimizer,
        scheduler,
        epochs,
        device,
        checkpoint_path
):
    best_val_loss = 1e6

    history = {}

    for epoch in range(0, epochs):

        train_loss, train_acc = train(
            net,
            train_dataloader,
            cls_criterion,
            reconstruction_criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate(
            net,
            val_dataloader,
            cls_criterion,
            reconstruction_criterion,
            device
        )
        scheduler.step()

        update_metrics(
            history,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc
        )

        if val_loss < best_val_loss:
            checkpoint(net, history, checkpoint_path, MODEL_NAME)
            best_val_loss = val_loss

    return history


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
