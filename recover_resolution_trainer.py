import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from data_processing.dataset import RecoverResolutionCifarDataset
from models.recover_resolution import ConvAutoEncoder
from basic_trainer import plot_training_loss, update_metrics, checkpoint

from utils.utils import progress_bar


def create_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Recover high resolution images')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--early_stopping_patience', default=10, type=int,
                        help='Early stopping patience')
    parser.add_argument('--img_input_size', default=8, type=int, help='Image Size')
    parser.add_argument('--img_output_size', default=32, type=int, help='Image Size')
    parser.add_argument('--cifar_data_path', required=True,
                        help='input_folder containing the CIFAR images')
    parser.add_argument('--checkpoint_path', required=True, help='checkpoint_path for the model')
    return parser


def create_datasets(args):
    dataset = RecoverResolutionCifarDataset(
        args.img_input_size,
        args.img_output_size,
        args.cifar_data_path
    )
    total = len(dataset)
    train_size = int(total * 0.9)
    val_size = total - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    return train_set, val_set


def main(args):
    # Data
    train_set, val_set = create_datasets(args)

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # net = create_recover_resolution_net(
    #     img_input_size=args.img_input_size,
    #     img_output_size=args.img_output_size
    # )

    net = ConvAutoEncoder()
    net = net.to(device)

    history = train_model(net, train_set, val_set, args, device)

    plot_training_loss(history, args.checkpoint_path)


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

    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=1e-4, eps=0.001
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    history = {}
    best_val_loss = 1e6
    for epoch in range(0, args.epochs):

        train_loss = train(net, train_dataloader, criterion, optimizer, device)
        val_loss = validate(net, val_dataloader, criterion, device)
        scheduler.step()

        update_metrics(
            history,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=0.0,
            val_acc=0.0
        )

        if val_loss < best_val_loss:
            checkpoint(net, history, args.checkpoint_path)
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Stop the training if the val does not improve
        if early_stopping_counter > args.early_stopping_patience:
            print("Validation loss has not improved in {} epochs, stopping early".format(
                args.early_stopping_patience))
            print("Obtained lowest validation loss of: {}".format(best_val_loss))
            break

    return history


def train(
        net,
        train_loader,
        criterion,
        optimizer,
        device
):
    net.train()
    train_loss = 0
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

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f' % (train_loss / (batch_idx + 1)))

    train_average_loss = train_loss / total

    return train_average_loss


def validate(
        net,
        val_loader,
        criterion,
        device
):
    net.eval()
    val_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            total += targets.size(0)

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f' % (val_loss / (batch_idx + 1)))

    val_average_loss = val_loss / total
    return val_average_loss


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
