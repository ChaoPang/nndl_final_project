import torch
from data_processing.dataset import ProjectDataSet
import argparse

# from models import *
parser = argparse.ArgumentParser(description='PyTorch NNDL project data')
parser.add_argument('--training_data_path', required=True,
                    help='input_folder containing the images')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')

args = parser.parse_args()

# Data
print('==> Preparing data..')
dataset = ProjectDataSet(
    image_folder_path=args.training_data_path
)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
h, w = 0, 0
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs = inputs.to(device)
    if batch_idx == 0:
        h, w = inputs.size(2), inputs.size(3)
        print(inputs.min(), inputs.max())
        chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
    else:
        chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
mean = chsum / len(dataset) / h / w
print('mean: %s' % mean.view(-1))

chsum = None
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs = inputs.to(device)
    if batch_idx == 0:
        chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    else:
        chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
std = torch.sqrt(chsum / (len(dataset) * h * w - 1))
print('std: %s' % std.view(-1))

print('Done!')
