"""Performs face alignment and stores face thumbnails in the output directory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import numpy as np
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
from torchvision import datasets, transforms
import torch

def create_output_class_dir(output_dir, class_names):
    for class_n in class_names:
        if not os.path.exists(os.path.join(output_dir, class_n)):
            os.makedirs(os.path.join(output_dir, class_n))


def get_file_with_parents(filepath, levels=1):
    common = filepath
    for i in range(levels + 1):
        common = os.path.dirname(common)
    return os.path.relpath(filepath, common)

def main(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Store some git revision info in a text file in the log directory
    trans = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(args.input_dir, transform=trans)
    create_output_class_dir(args.output_dir, dataset.classes)

    print('Creating networks and loading parameters')

    minsize = 10  # minimum size of face
    threshold = [0.5, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 40

    mtcnn = MTCNN(
        image_size=160, margin=margin, min_face_size=minsize,
        thresholds=threshold, factor=factor, post_process=False,
        device=device
    )

    for img_path, label in dataset.samples:
        img_class_path_name = get_file_with_parents(img_path, 1)

        srcBGR = cv2.imread(img_path)
        destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(destRGB)

        new_path = os.path.join(args.output_dir, img_class_path_name)

        mtcnn(img, save_path=new_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))