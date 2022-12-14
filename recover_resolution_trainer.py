import argparse

from torch.utils.data import DataLoader
from data_processing.recover_dataset import RecoverResolutionDataset
from models.recover_resolution import *
from basic_trainer import plot_training_loss, update_metrics, checkpoint

from utils.utils import progress_bar


def create_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Recover high resolution images')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--img_input_size', default=8, type=int, help='Image Size')
    parser.add_argument('--img_output_size', default=32, type=int, help='Image Size')
    parser.add_argument('--data_path', required=True,
                        help='input_folder containing the CIFAR images')
    parser.add_argument('--checkpoint_path', required=True, help='checkpoint_path for the model')
    return parser


def main(args):
    # Data
    dataset = RecoverResolutionDataset(
        args.img_input_size,
        args.img_output_size,
        args.data_path
    )

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # net = create_recover_resolution_net(
    #     img_input_size=args.img_input_size,
    #     img_output_size=args.img_output_size
    # )

    # net = ConvAutoEncoder()
    net = ConvAutoEncoderV2()
    # net = SubPixelCNN()
    net = net.to(device)

    history = train_model(net, dataset, args, device)

    plot_training_loss(history, args.checkpoint_path)


def train_model(
        net,
        train_set,
        args,
        device
):
    train_dataloader = DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=4
    )

    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=1e-4, eps=0.001
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    history = {}
    for epoch in range(0, args.epochs):
        train_loss = train(net, train_dataloader, criterion, optimizer, device)
        scheduler.step()

        update_metrics(
            history,
            train_loss=train_loss,
            val_loss=0.0,
            train_acc=0.0,
            val_acc=0.0
        )
        checkpoint(net, history, args.checkpoint_path, f'model-{epoch}.pt')

    # Save for the last time
    checkpoint(net, history, args.checkpoint_path, 'final_model.pt')

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


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
