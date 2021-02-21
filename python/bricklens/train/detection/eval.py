#!/usr/bin/env python3

from __future__ import division

import argparse
import datetime
import gc
import os
import sys
import time
import tracemalloc

import numpy as np
import torch
import torch.optim as optim
import yaml
from PIL import Image
from pympler import muppy, summary, tracker
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataset
import utils
from model import Darknet

# Get command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_config_path",
    type=str,
    default="bricklens/train/detection/config/yolov3.yml",
    help="Path to model config file",
)
parser.add_argument(
    "--dataset_config_path",
    type=str,
    default="bricklens/train/detection/config/dataset.yml",
    help="Path to dataset config file",
)
parser.add_argument(
    "--weights_path",
    type=str,
    required=True,
    help="Path to model weights file",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=0,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--num_images",
    type=int,
    default=10,
    help="number of images to process",
)
parser.add_argument(
    "--use_cuda", type=bool, default=True, help="whether to use cuda if available"
)
parser.add_argument(
    "--obj_threshold",
    type=float,
    default=0.25,
    help="Objectness threshold for each bounding box",
)
args = parser.parse_args()
print(args)

# Check for CUDA.
cuda = torch.cuda.is_available() and args.use_cuda
print(f"Using CUDA: {cuda}")
device = "cuda:0" if cuda else "cpu"

# Get dataset configuration.
with open(args.dataset_config_path, "r") as stream:
    data_config = yaml.safe_load(stream)
data_path = data_config["train"]  # XXX MDW - For testing, should be 'val'
classfile_path = data_config["classes"]

# Get model parameters.
with open(args.model_config_path, "r") as stream:
    model_config = yaml.safe_load(stream)

# Create model.
model = Darknet(model_config, training=False)
model.load_weights(args.weights_path)
if cuda:
    model = model.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Create dataloader.
dset = dataset.ListDataset(data_path, classfile_path)
dataloader = torch.utils.data.DataLoader(
    dset,
    batch_size=1,
    shuffle=False,
    num_workers=args.n_cpu,
)


for index, (fname, imgs, targets) in enumerate(dataloader):
    assert len(fname) == len(imgs) == len(targets) == 1

    if index > args.num_images:
        break

    input_image = np.array(Image.open(fname[0]))
    assert imgs.size(2) == imgs.size(3)  # Expect square image from the dataloader.
    img_dim = imgs.size(2)

    print(f"Input index [{index}]: {fname[0]}")
    print(f"  Image has size {imgs[0].size()}")
    print(f"  Target has size {targets[0].size()}")

    imgs = Variable(imgs.type(Tensor))
    targets = Variable(targets.type(Tensor), requires_grad=False)
    with torch.no_grad():
        detections = model(imgs, targets)
        detections = utils.non_max_suppression(detections, num_classes=40, conf_thres=args.obj_threshold, nms_thres=0.4)[0]
    print(f"MDW: detections is: {detections}")
    if detections is None:
        print(f"WARNING: No detections for {fname[0]}")
        continue

    # Rescale detection bboxes to size of original input image.
    detections = utils.rescale_boxes(detections, img_dim, input_image.shape[:2])
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        topclass = dset.classindex_to_name(cls_pred)
        print(f"MDW: bbox conf {conf} class {topclass} cls_conf {cls_conf} = ({x1},{y1})..({x2},{y2})")
