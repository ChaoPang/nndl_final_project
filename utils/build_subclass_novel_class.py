import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image

from data_processing.recover_dataset import CIFAR100Coarse


def get_novel_class(data_folder, train):
    # Download the CIFAR100 Data
    cifar100_data = CIFAR100Coarse(
        root=data_folder,
        train=train,
        download=True
    )

    images = cifar100_data.data
    labels = np.asarray(cifar100_data.targets)

    # Filter out all reptile images
    non_reptile_images = images[np.squeeze(labels != 15)]

    random_indices = np.random.choice(
        len(non_reptile_images),
        size=200,
        replace=False
    )

    random_images = non_reptile_images[random_indices]

    return random_images


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Download images for sub classes')
    parser.add_argument('--training_label_path', required=True, help='the path to training label')
    parser.add_argument('--new_train_label_path', required=True, help='The output_folder')
    parser.add_argument('--cifar_data_folder', required=True, help='The output_folder')
    parser.add_argument('--image_folder', required=True, help='The output_folder')
    parser.add_argument('--is_train', action='store_true')
    return parser


def main(args):
    subclass_mapping = pd.read_csv(args.training_label_path, header=0)

    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)

    last_image_number = max(
        [int(item[1][:-4]) for item in subclass_mapping.image.iteritems()]
    ) + 1

    novel_images = get_novel_class(args.cifar_data_folder, args.is_train)

    new_images = []
    for img_array in novel_images:
        img = Image.fromarray(img_array)
        image_name = f'{last_image_number}.jpg'
        img.save(os.path.join(args.image_folder, image_name))
        last_image_number += 1
        # 89 is the novel index
        if args.is_train:
            new_images.append((image_name, 3, 89))
        else:
            new_images.append((image_name, 89))

    pd.concat(
        [subclass_mapping,
         pd.DataFrame(new_images, columns=subclass_mapping.columns)]
    ).to_csv(args.new_train_label_path, index=False)


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
