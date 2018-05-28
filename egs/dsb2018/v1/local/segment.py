#!/usr/bin/env python3

import torch
import argparse
import os
import sys
import random
import numpy as np
import scipy.misc
from models.Unet import UNet
from train import sample
from waldo.segmenter import ObjectSegmenter, SegmenterOptions
from waldo.core_config import CoreConfig
from waldo.data_visualization import visualize_mask
from waldo.data_io import WaldoDataset
from unet_config import UnetConfig


parser = argparse.ArgumentParser(description='Pytorch DSB2018 setup')
parser.add_argument('test_data', default='./data', type=str,
                    help='Path to processed validation data')
parser.add_argument('dir', type=str,
                    help='Directory to store segmentation results. '
                    'It is assumed that <dir> is a sub-directory of '
                    'the model directory.')
parser.add_argument('--model', type=str, default='model_best.pth.tar',
                    help='Name of the model file to use for segmenting.')
parser.add_argument('--train-image-size', default=128, type=int,
                    help='The size of the parts of training images that we'
                    'train on (in order to form a fixed minibatch size).'
                    'These are derived from the input images'
                    ' by padding and then random cropping.')
parser.add_argument('--object-merge-factor', type=float, default=None,
                    help='Scale for object merge scores in the segmentaion '
                    'algorithm. If not set, it will be set to '
                    '1.0 / num_offsets by default.')
parser.add_argument('--same-different-bias', type=float, default=0.0,
                    help='Bias for same/different probs in the segmentation '
                    'algorithm.')
random.seed(0)
np.random.seed(0)


def main():
    global args
    args = parser.parse_args()
    args.batch_size = 1  # only segment one image for experiment

    model_dir = os.path.dirname(args.dir)
    core_config_path = os.path.join(model_dir, 'configs/core.config')
    unet_config_path = os.path.join(model_dir, 'configs/unet.config')

    core_config = CoreConfig()
    core_config.read(core_config_path)
    print('Using core configuration from {}'.format(core_config_path))

    # loading Unet configuration
    unet_config = UnetConfig()
    unet_config.read(unet_config_path, args.train_image_size)
    print('Using unet configuration from {}'.format(unet_config_path))

    offset_list = core_config.offsets
    print("offsets are: {}".format(offset_list))

    # model configurations from core config
    num_classes = core_config.num_classes
    num_colors = core_config.num_colors
    num_offsets = len(core_config.offsets)
    # model configurations from unet config
    start_filters = unet_config.start_filters
    up_mode = unet_config.up_mode
    merge_mode = unet_config.merge_mode
    depth = unet_config.depth

    model = UNet(num_classes, num_offsets,
                 in_channels=num_colors, depth=depth,
                 start_filts=start_filters,
                 up_mode=up_mode,
                 merge_mode=merge_mode)

    model_path = os.path.join(model_dir, args.model)
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("loaded.")
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    model.eval()  # convert the model into evaluation mode

    testset = WaldoDataset(args.test_data, core_config, args.train_image_size)
    print('Total samples in the test set: {0}'.format(len(testset)))

    dataloader = torch.utils.data.DataLoader(
        testset, num_workers=1, batch_size=args.batch_size)

    segment_dir = args.dir
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)
    img, class_pred, adj_pred = sample(
        model, dataloader, segment_dir, core_config)

    if args.object_merge_factor is None:
        args.object_merge_factor = 1.0 / len(offset_list)
    segmenter_opts = SegmenterOptions(same_different_bias = args.same_different_bias,
                                      object_merge_factor = args.object_merge_factor)
    seg = ObjectSegmenter(class_pred[0].detach().numpy(),
                          adj_pred[0].detach().numpy(),
                          num_classes, offset_list, segmenter_opts)
    mask_pred, object_class = seg.run_segmentation()
    x = {}
    # from (color, height, width) to (height, width, color)
    x['img'] = np.moveaxis(img[0].numpy(), 0, -1)
    x['mask'] = mask_pred.astype(int)
    x['object_class'] = object_class
    visualize_mask(x, core_config)
    scipy.misc.imsave('{}/final.png'.format(segment_dir), x['img_with_mask'])


if __name__ == '__main__':
    main()
