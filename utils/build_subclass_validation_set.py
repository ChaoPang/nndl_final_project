import os
import argparse
import pandas as pd
from serpapi import GoogleSearch

from PIL import Image
import requests
from io import BytesIO


def download_images(
        query: str,
        output_folder: str,
        file_name_base: int = 0
):
    search = GoogleSearch({
        "q": query,
        "tbm": "isch",
        'async': 'false',
        "api_key": "50d03ab632fea45434331d4361e3843217a0630281985e3f3ea31a3af2ef15da"
    })

    image_names = []
    counter = 0
    for image_result in search.get_dict()['images_results']:
        link = image_result["thumbnail"]
        try:
            print("link: " + link)
            response = requests.get(link)
            img = Image.open(BytesIO(response.content))
            image_name = f'{file_name_base + counter}.jpg'
            image_names.append(image_name)
            img.save(os.path.join(output_folder, image_name))
        except:
            print(f'Failed to download {link}')
        counter += 1

    return image_names


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Download images for sub classes')
    parser.add_argument('--subclass_mapping_file', dest='subclass_mapping_file', required=True,
                        help='The path to the subclass mapping file')
    parser.add_argument('--output_folder', dest='output_folder', required=True,
                        help='The output_folder')
    return parser


def main(args):
    subclass_mapping = pd.read_csv(args.subclass_mapping_file, header=0)
    val_image_folder = os.path.join(args.output_folder, 'val_images')

    if not os.path.exists(val_image_folder):
        os.makedirs(val_image_folder)

    file_name_base = 0
    image_labels = []
    for t in subclass_mapping.itertuples():
        class_index = t[1]
        class_name = t[2]
        class_name = class_name.split(',')[0]

        if class_name == 'novel':
            continue
        image_names = download_images(
            class_name,
            val_image_folder,
            file_name_base
        )

        file_name_base += len(image_names)
        image_labels.extend([(n, class_index) for n in image_names])

    image_labels_pd = pd.DataFrame(image_labels, columns=['image', 'subclass_index'])
    image_labels_pd.to_csv(os.path.join(args.output_folder, 'val_data.csv'), index=False)


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
