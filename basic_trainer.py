import os
import pickle
import argparse
import numpy as np
import pandas as pd

import torch.optim as optim
from torch.utils.data import DataLoader

from data_processing.dataset import ProjectDataSet, ExtractedCifarDataset
from models.resnet import *
from models.finetune_pretrained import FinetuneResnet152, FinetuneRegNet

from utils.utils import progress_bar
import matplotlib.pyplot as plt

MODEL_NAME = 'best_model.pt'

IDX_TO_SUPERCLASS_DICT = {
    0: 'bird',
    1: 'dog',
    2: 'reptile'
}


def map_idx_to_superclass(
        predictions: pd.Series
):
    return predictions.apply(IDX_TO_SUPERCLASS_DICT.get)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch NNDL image classification challenge')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--early_stopping_patience', default=10, type=int,
                        help='Early stopping patience')
    parser.add_argument('--img_size', default=8, type=int, help='Image Size')
    parser.add_argument('--training_data_path', required=True,
                        help='input_folder containing the images')
    parser.add_argument('--training_label_path', required=True, help='the path to training label')
    parser.add_argument('--test_data_path', required=True,
                        help='input_folder containing the images')
    parser.add_argument('--cifar_data_path', required=False,
                        help='input_folder containing the CIFAR images')
    parser.add_argument('--checkpoint_path', required=True, help='checkpoint_path for the model')
    parser.add_argument('--external_validation', action='store_true',
                        help='Using CIFAR data to test the model')
    parser.add_argument('--test_label', action='store_true',
                        help='Indicate whether the test label is available')
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
        test_set,
        device,
        is_label_available: bool = False
):
    data_loader = DataLoader(
        test_set, batch_size=128, num_workers=4
    )

    net.eval()
    predictions = []
    labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = net(inputs.to(device))
            predicted = torch.argmax(outputs, dim=-1)
            predictions.append(predicted.detach().cpu().numpy())
            if is_label_available:
                labels.append(targets.detach().cpu().numpy())
                total += targets.size(0)
                correct += predicted.detach().cpu().eq(targets).sum().item()
                progress_bar(batch_idx, len(data_loader), 'Acc: %.3f%% (%d/%d)'
                             % (100. * correct / total, correct, total))
    predictions_pd = pd.DataFrame(np.hstack(predictions), columns=['predictions'])
    predictions_pd['prediction_class'] = map_idx_to_superclass(predictions_pd.predictions)

    if is_label_available:
        predictions_pd['label'] = np.hstack(labels)

    return predictions_pd


def train_model(
        net,
        train_set,
        val_set,
        args,
        device
):
    train_dataloader = DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=4
    )

    val_dataloader = DataLoader(
        val_set, batch_size=128, shuffle=True, num_workers=4
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=1e-4, eps=0.1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    history = training_loop(
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

    return history


def main(args):
    # Data

    train_set, val_set, test_set = create_datasets(args)

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # net = ResNet101(num_classes=3)
    # net = FinetuneResnet152(num_classes=3)
    net = FinetuneRegNet(num_classes=3)
    net = net.to(device)

    history = train_model(net, train_set, val_set, args, device)

    net = torch.load(os.path.join(args.checkpoint_path, MODEL_NAME))
    predictions_pd = predict(net, test_set, device, args.test_label)
    predictions_pd.to_csv(
        os.path.join(args.checkpoint_path, 'predictions.csv'),
        index=False
    )

    # Plot training curve
    plt.figure()
    plt.plot(history['train_loss'], "ro-", label="Train")
    plt.plot(history['val_loss'], "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig(args.checkpoint_path + "/training_curve.png")


def create_datasets(
        args
):
    training_dataset = ProjectDataSet(
        image_folder_path=args.training_data_path,
        data_label_path=args.training_label_path,
        is_training=True,
        is_superclass=True,
        img_size=args.img_size
    )

    if args.external_validation and args.cifar_data_path:
        # Use the CIFAR data as the external validation set
        cifar_train_set = ExtractedCifarDataset(
            args.cifar_data_path,
            train=True,
            img_size=args.img_size
        )
        cifar_test_set = ExtractedCifarDataset(
            args.cifar_data_path,
            train=False,
            img_size=args.img_size
        )
        test_set = torch.utils.data.ConcatDataset(
            [cifar_train_set, cifar_test_set])
    else:

        if args.cifar_data_path:
            cifar_train_set = ExtractedCifarDataset(
                args.cifar_data_path,
                train=True,
                img_size=args.img_size
            )
            cifar_test_set = ExtractedCifarDataset(
                args.cifar_data_path,
                train=False,
                img_size=args.img_size
            )
            training_dataset = torch.utils.data.ConcatDataset(
                [training_dataset, cifar_train_set, cifar_test_set])

        test_set = ProjectDataSet(
            image_folder_path=args.test_data_path,
            is_training=False,
            is_superclass=True,
            img_size=args.img_size
        )

    train_total = len(training_dataset)
    train_size = int(train_total * 0.9)
    val_size = train_total - train_size
    train_set, val_set = torch.utils.data.random_split(
        training_dataset, [train_size, val_size]
    )

    return train_set, val_set, test_set


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
