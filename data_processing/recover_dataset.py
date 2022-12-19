import os
import numpy as np
import torch.utils
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, Caltech101, Caltech256, ImageFolder

IMAGENET_FOLDER = 'imagenet'


class RecoverResolutionDataset(Dataset):

    def __init__(
            self,
            img_input_size=8,
            img_output_size=32,
            data_folder='./data',
            download=True
    ):
        self._img_input_size = img_input_size
        self._img_output_size = img_output_size
        self._data_folder = data_folder
        self._download = download
        self._cifar_dataset = torch.utils.data.ConcatDataset([
            # self._get_cifar10_dataset(),
            # self._get_cifar100_dataset(),
            self._get_imagenet_dataset()
            # self._get_caltech101_dataset(),
            # self._get_caltech256_dataset()
        ])
        self._resize_input_transformers = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self._img_input_size, self._img_input_size)),
            torchvision.transforms.ToTensor(),
        ])
        self._resize_output_transformers = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self._img_output_size, self._img_output_size)),
            torchvision.transforms.ToTensor(),
        ])

    def _get_cifar10_dataset(self):
        cifar10_train = CIFAR10(
            root=self._data_folder,
            train=True,
            download=self._download
        )
        cifar10_test = CIFAR10(
            root=self._data_folder,
            train=False,
            download=self._download
        )
        return torch.utils.data.ConcatDataset([
            cifar10_train, cifar10_test
        ])

    def _get_cifar100_dataset(self):
        cifar100_train = CIFAR100Coarse(
            root=self._data_folder,
            train=True,
            download=self._download
        )
        cifar100_test = CIFAR100Coarse(
            root=self._data_folder,
            train=False,
            download=self._download
        )
        return torch.utils.data.ConcatDataset([
            cifar100_train, cifar100_test
        ])

    def _get_imagenet_dataset(self):
        transformers = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self._img_output_size, self._img_output_size))
        ])
        imagenet = ImageFolder(
            root=os.path.join(self._data_folder, IMAGENET_FOLDER),
            transform=transformers
        )
        return imagenet

    def _get_caltech101_dataset(self):
        transformers = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
            torchvision.transforms.Resize((self._img_output_size, self._img_output_size))
        ])
        caltech101 = Caltech101(
            root=self._data_folder,
            download=self._download,
            transform=transformers
        )
        return caltech101

    def _get_caltech256_dataset(self):
        transformers = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
            torchvision.transforms.Resize((self._img_output_size, self._img_output_size))
        ])
        caltech256 = Caltech256(
            root=self._data_folder,
            download=self._download,
            transform=transformers
        )
        return caltech256

    def __getitem__(self, index):
        img, _ = self._cifar_dataset[index]
        return self._resize_input_transformers(img), self._resize_output_transformers(img)

    def __len__(self):
        return len(self._cifar_dataset)


# Credits goes to https://github.com/ryanchankh/cifar100coarse
class CIFAR100Coarse(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
