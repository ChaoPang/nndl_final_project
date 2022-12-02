import sys
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional

import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from data_processing.dataset import ProjectDataSet, CifarValidationDataset, get_data_normalize
from models.finetune_pretrained import *

from utils.utils import progress_bar
from utils.compute_mean_std import calculate_stats

import matplotlib.pyplot as plt

MODEL_NAME = 'best_model.pt'

IDX_TO_SUPERCLASS_DICT = {
    0: 'bird',
    1: 'dog',
    2: 'reptile'
}

sys.setrecursionlimit(10000)

pretrained_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def map_idx_to_superclass(
        predictions: pd.Series
):
    return predictions.apply(IDX_TO_SUPERCLASS_DICT.get)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch NNDL image classification challenge')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='Dropout rate')
    parser.add_argument('--freeze_weight', action='store_true',
                        help='Whether or not we freeze the weights of the pretrained model')
    parser.add_argument('--deep_feature', action='store_true',
                        help='Whether or not extract the deep feature')
    parser.add_argument('--early_stopping_patience', default=10, type=int,
                        help='Early stopping patience')
    parser.add_argument('--img_size', default=8, type=int, help='Image Size')
    parser.add_argument('--training_data_path', required=True,
                        help='input_folder containing the images')
    parser.add_argument('--training_label_path', required=True, help='the path to training label')
    parser.add_argument('--test_data_path', required=True,
                        help='input_folder containing the images')
    parser.add_argument('--cifar_data_path', required=True,
                        help='input_folder containing the CIFAR images')
    parser.add_argument('--checkpoint_path', required=True, help='checkpoint_path for the model')
    parser.add_argument('--external_validation', action='store_true',
                        help='Using CIFAR data to test the model')
    parser.add_argument('--test_label', action='store_true',
                        help='Indicate whether the test label is available')
    parser.add_argument('--up_sampler_path', required=False,
                        help='Path to the up sampler')
    parser.add_argument('--img_upsampled_size', required='--up_sampler_path' in sys.argv, type=int,
                        help='img upsampled size by auto encoder')

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
        criterion,
        optimizer,
        device,
        up_sampler: nn.Module = None,
):
    if up_sampler:
        up_sampler.eval()

    net.train()

    train_loss = 0
    correct = 0
    total = 0

    # Get normalize transform
    if up_sampler:
        train_mean, train_std = calculate_stats(train_loader)
        data_normalize_transform = transforms.Normalize(
            mean=train_mean,
            std=train_std
        )
    else:
        data_normalize_transform = get_data_normalize()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        if up_sampler:
            inputs = up_sampler(inputs)

        inputs = data_normalize_transform(inputs)

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
        is_label_available: bool = False,
        up_sampler: nn.Module = None
):
    data_loader = DataLoader(
        test_set, batch_size=128, num_workers=4
    )

    # Get normalize transform
    if up_sampler:
        train_mean, train_std = calculate_stats(data_loader)
        data_normalize_transform = transforms.Normalize(
            mean=train_mean,
            std=train_std
        )
    else:
        data_normalize_transform = get_data_normalize()

    net.eval()
    predictions = []
    labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = data_normalize_transform(inputs)
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
        device,
        up_sampler: nn.Module = None
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
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
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
        args.checkpoint_path,
        up_sampler
    )

    return history


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    train_set, val_set, test_set = create_datasets(args)

    # Initialize the model
    device = get_device()

    if args.up_sampler_path:
        up_sampler = torch.load(
            args.up_sampler_path,
            map_location=device
        )
    else:
        up_sampler = None

    # net = FinetuneResnet152(num_classes=3, deep_feature=args.deep_feature)
    # net = FinetuneRegNet(num_classes=3, deep_feature=args.deep_feature)
    # net = FinetuneEfficientNetV2(num_classes=3, deep_feature=args.deep_feature)
    net = FinetuneEfficientNetEnsembleModel(
        num_classes=3,
        dropout_rate=args.dropout_rate,
        freeze_weight=args.freeze_weight,
        device=device,
        deep_feature=args.deep_feature
    )
    net = net.to(device)

    history = train_model(net, train_set, val_set, args, device, up_sampler)

    net = torch.load(os.path.join(args.checkpoint_path, MODEL_NAME))
    predictions_pd = predict(net, test_set, device, args.test_label)
    predictions_pd.to_csv(
        os.path.join(args.checkpoint_path, 'predictions.csv'),
        index=False
    )

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
    train_set = ProjectDataSet(
        image_folder_path=args.training_data_path,
        data_label_path=args.training_label_path,
        is_training=True,
        is_superclass=True,
        img_size=args.img_size,
        normalize=False
    )

    test_set = ProjectDataSet(
        image_folder_path=args.test_data_path,
        is_training=False,
        is_superclass=True,
        img_size=args.img_size,
        normalize=False
    )

    # If the up sampler is enabled, we use the default 32 by 32 image for validation
    # Use the CIFAR data as the external validation set
    val_set = CifarValidationDataset(
        cifar_data_folder=args.cifar_data_path,
        download=True,
        img_size=args.img_upsampled_size if args.up_sampler_path else args.img_size
    )

    if not args.external_validation:
        train_set = torch.utils.data.ConcatDataset(
            [train_set, val_set])
        train_total = len(train_set)
        train_size = int(train_total * 0.8)
        val_size = train_total - train_size
        train_set, val_set = torch.utils.data.random_split(
            train_set, [train_size, val_size]
        )

    print(f'train_set size: {len(train_set)}')
    print(f'val_set size: {len(val_set)}')
    print(f'test_set size: {len(test_set)}')

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
        val_dataloader,
        criterion,
        optimizer,
        scheduler,
        epochs,
        device,
        early_stopping_patience,
        checkpoint_path,
        up_sampler: nn.Module = None
):
    early_stopping_counter = 0
    best_val_loss = 1e6

    history = {}

    for epoch in range(0, epochs):

        train_loss, train_acc = train(
            net,
            train_dataloader,
            criterion,
            optimizer,
            device,
            up_sampler
        )
        val_loss, val_acc = validate(
            net,
            val_dataloader,
            criterion,
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
