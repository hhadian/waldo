import torch
import argparse
import os
import sys
import torchvision
import random
from torchvision import transforms as tsf
from models.Unet import UNet
from dataset import Dataset_dsb2018
from waldo.segmenter import ObjectSegmenter
from waldo.core_config import CoreConfig
from unet_config import UnetConfig


parser = argparse.ArgumentParser(description='Pytorch DSB2018 setup')
parser.add_argument('model_dir', type=str,
                    help='path to the model dir')
parser.add_argument('--val-data', default='./data/val.pth.tar', type=str,
                    help='Path of processed validation data')
parser.add_argument('--test-data', default='./data/test.pth.tar', type=str,
                    help='Path of processed test data')

random.seed(0)


def main():
    global args
    args = parser.parse_args()
    args.batch_size = 1
    core_config_path = os.path.join(args.model_dir, 'configs/core.config')
    unet_config_path = os.path.join(args.model_dir, 'configs/unet.config')

    core_config = CoreConfig()
    core_config.read(core_config_path)
    print('Using core configuration from {}'.format(core_config_path))

    # loading Unet configuration
    unet_config = UnetConfig()
    unet_config.read(unet_config_path, core_config)
    print('Using unet configuration from {}'.format(unet_config_path))

    offset_list = core_config.offsets
    print("offsets are: {}".format(offset_list))

    # model configurations from core config
    image_width = core_config.train_image_size
    image_height = core_config.train_image_size
    num_classes = core_config.num_classes
    num_colors = core_config.num_colors
    num_offsets = len(core_config.offsets)


    # # of classes, # of offsets
    model = UNet(num_classes, num_offsets)

    model_path = os.path.join(args.model_dir, 'model_best.pth.tar')
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("loaded.")
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    s_trans = tsf.Compose([
        tsf.ToPILImage(),
        tsf.Resize((image_width, image_height)),
        tsf.ToTensor(),
    ])

    testset = Dataset_dsb2018(args.val_data, s_trans, offset_list,
                              num_classes, image_height, image_width)
    print('Total samples in the test set: {0}'.format(len(testset)))

    dataloader = torch.utils.data.DataLoader(
        testset, num_workers=1, batch_size=args.batch_size)

    data_iter = iter(dataloader)
    # data_iter.next()
    img, class_id, sameness = data_iter.next()
    torch.set_printoptions(threshold=5000)
    torchvision.utils.save_image(img, 'input.png')
    torchvision.utils.save_image(sameness[0, 0, :, :], 'sameness0.png')
    torchvision.utils.save_image(sameness[0, 1, :, :], 'sameness1.png')
    torchvision.utils.save_image(
        class_id[0, 0, :, :], 'class0.png')  # backgrnd
    torchvision.utils.save_image(class_id[0, 1, :, :], 'class1.png')  # cells

    model.eval()  # convert the model into evaluation mode

    #img = torch.autograd.Variable(img)
    predictions = model(img)
    # [batch-idx, class-idx, row, col]
    class_pred = predictions[0, :num_classes, :, :]
    # [batch-idx, offset-idx, row, col]
    adj_pred = predictions[0, num_classes:, :, :]

    for i in range(len(offset_list)):
        torchvision.utils.save_image(
            adj_pred[i, :, :], 'sameness_pred{}.png'.format(i))
    for i in range(num_classes):
        torchvision.utils.save_image(
            class_pred[i, :, :], 'class_pred{}.png'.format(i))

    seg = ObjectSegmenter(class_pred.detach().numpy(),
                          adj_pred.detach().numpy(), num_classes, offset_list)
    seg.run_segmentation()

    for i in range(len(offset_list)):
        torchvision.utils.save_image(
            adj_pred[i, :, :], 'sameness_pred{}.png'.format(i))
    for i in range(num_classes):
        torchvision.utils.save_image(
            class_pred[i, :, :], 'class_pred{}.png'.format(i))


if __name__ == '__main__':
    main()
