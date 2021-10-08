import argparse
import os
import pickle
import time
import random

import cv2
import colorsys

import faiss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import clustering


IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD  = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]


class VGG19(nn.Module):
    def __init__(self, net='pytorch', normalize_inputs=True):
        super(VGG19, self).__init__()

        self.normalize_inputs = normalize_inputs
        self.mean_ = IMAGENET_MEAN
        self.std_ = IMAGENET_STD

        vgg = torchvision.models.vgg19(pretrained=True).features

        for weights in vgg.parameters():
            weights.requires_grad = False

        vgg_avg_pooling = []
        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=1, stride=1, padding=0))
            else:
                vgg_avg_pooling.append(module)

        self.vgg = nn.Sequential(*vgg_avg_pooling[:-2])

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def forward(self, input):
        if self.normalize_inputs:
            input = self.do_normalize_inputs(input)
        output = self.vgg(input)
        return output


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(float(i) / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def denormalize_inputs(x):
    return x * IMAGENET_STD.to(x.device) + IMAGENET_MEAN.to(x.device)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--data',       type=str, help='path to dataset')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'], default='Kmeans')
    parser.add_argument('--num_cluster', '--k', type=int, default=15)
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    
    parser.add_argument('--batch', default=10, type=int)
    parser.add_argument('--workers', default=4, type=int)

    return parser.parse_args()


def main(args):
    print('-------------- args --------------')
    print(args)


    print('-------------- MODEL --------------')
    model = VGG19().cuda()
    print(model)


    print('------------- ImageNet -------------')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    end = time.time()
    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
    print('Load dataset: {0:.2f} s'.format(time.time() - end))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)


    print('-------------- feature infer -----------')  

    for i, (input_tensor, _) in enumerate(dataloader):
        with torch.no_grad():
            input_var = input_tensor.cuda()
            features  = model.forward(input_var)

            print(  'inputs:', input_var.shape, torch.min(input_var), torch.max(input_var))
            print('features:', features.shape)

        break

    b,c,h,w = input_var.size()
    images  = 255 * denormalize_inputs(input_var).permute(2,0,3,1).contiguous() # H x B x W x C
    images  = images.view(h, -1, c).cpu().numpy().astype(np.uint8)

    b,c,h,w  = features.size()
    features = features.view(b, c, -1).transpose(1,2).contiguous() # B x H*W x C
    
    xv, yv = torch.meshgrid([torch.arange(0,h), torch.arange(0,w)]) # H x W , H x W
    xv = xv.view(1, h, w, 1) 
    yv = yv.view(1, h, w, 1)
    coords = torch.cat([xv,yv], dim=3).cuda() # 1 x H x W x 2 
    coords = coords.expand(size=(b,h,w,2)).view(b,-1, 2).contiguous().float() # B x H*W x 2

    coords = coords / (h*w)

    print('  images:', images.shape)
    print('features:', features.shape)
    print('  coords:', coords.shape)

    #features = torch.cat([features, coords], dim=-1)

    print('  concat:', features.shape)


    print('-------------- Clustering -----------')

    segmentations = []
    for idx in range(b):
        feature = features[idx].cpu().numpy().astype(np.float32)
        #coord   = coords[idx].cpu().numpy()
        coord   = None

        clusterizer = clustering.__dict__[args.clustering](args.num_cluster)
        I, loss     = clusterizer.cluster(data=feature, 
                                          coord_grid=coord, 
                                          verbose=True)
        cluster_map = np.array(I).reshape(h,w)

        segmentation = np.zeros(shape=(h,w,3))
        lbl_colors   = random_colors(args.num_cluster)
        for j, color in enumerate(lbl_colors):
            segmentation[cluster_map==j] = color
        segmentation = (255*segmentation).astype(np.uint8)

        print('segmentation:', segmentation.shape)
        segmentations.append(segmentation)

    segmentations = np.concatenate(segmentations, axis=1) # along Width

    print('segmentations:', segmentations.shape)


    result = np.concatenate([images,segmentations], axis=0)
      
    cv2.imwrite(f'batch0_img.png', result)
    

if __name__ == '__main__':
    args = parse_args()
    main(args=args)
