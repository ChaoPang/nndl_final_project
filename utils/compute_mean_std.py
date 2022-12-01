import torch
from data_processing.dataset import ProjectDataSet
import argparse


def calculate_stats(
        dataloader
):
    # Calculate mean
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h, w = 0, 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        if batch_idx == 0:
            h, w = inputs.size(2), inputs.size(3)
            chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
    mean = chsum / len(dataloader.dataset) / h / w

    # Calculate std
    chsum = None
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        if batch_idx == 0:
            chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    std = torch.sqrt(chsum / (len(dataloader.dataset) * h * w - 1))

    return torch.squeeze(mean).detach().cpu().numpy(), torch.squeeze(std).detach().cpu().numpy()


def create_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch NNDL project data')
    parser.add_argument('--training_data_path', required=True,
                        help='input_folder containing the images')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    return parser.parse_args()


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    print('==> Preparing data..')
    dataset = ProjectDataSet(
        image_folder_path=args.training_data_path
    )
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=2)
    mean, std = calculate_stats(trainloader)
    print('mean: %s' % mean.view(-1))
    print('std: %s' % std.view(-1))
    print('Done!')
