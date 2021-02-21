#!/usr/bin/env python3

from __future__ import division

import argparse
import datetime
import gc
import os
import sys
import time
import tracemalloc

import torch
import torch.optim as optim
import yaml
from pympler import muppy, summary, tracker
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataset
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
    assert len(fname) == len(imgs) == len(targets)

    # XXX MDW HACKING
    if not fname[0].endswith("image_00000.png"):
        continue

    batch_size = len(imgs)
    img_width = imgs.size(2)
    img_height = imgs.size(3)

    print(f"Input index [{index}]: {fname[0]}")
    for batch_index in range(batch_size):
        print(f"  Image [{batch_index}] has size {imgs[batch_index].size()}")
        print(f"  Target [{batch_index}] has size {targets[batch_index].size()}")

    imgs = Variable(imgs.type(Tensor))
    targets = Variable(targets.type(Tensor), requires_grad=False)
    labels = model(imgs, targets)

    print(f"Output is of length {len(labels)}")
    for ix, label in enumerate(labels):
        pred_boxes, pred_conf, pred_cls = label
        print(f"MDW: label[{ix}] pred_boxes.size is {pred_boxes.size()}")
        boxes = pred_boxes.view(batch_size, -1, 4)
        conf = pred_conf.view(batch_size, -1, 1)
        classes = pred_cls.view(batch_size, -1, dset.num_classes)
        print(f"MDW: label[{ix}] boxes.size is {boxes.size()}")
        print(f"MDW: label[{ix}] conf.size is {conf.size()}")
        print(f"MDW: label[{ix}] classes.size is {classes.size()}")

        # Use to map bbox coordinates back to pixels.
        #grid_cells_x = pred_boxes.size(2)
        #grid_cells_y = pred_boxes.size(3)

        for batch_index in range(batch_size):
            for box_index in range(boxes.size(1)):
                box = boxes[batch_index, box_index]
                boxconf = conf[batch_index, box_index, 0]
                classconf = classes[batch_index, box_index, ...]
                if boxconf < args.obj_threshold:
                    continue

                topclass = torch.argmax(classconf).item()
                topclass_conf = classconf[topclass]
                topclass = dset.classindex_to_name(topclass)

                # Generate (upper left, lower right) from raw prediction.
                #x1 = box[0] - (box[2] / 2)
                #y1 = box[1] - (box[3] / 2)
                #x2 = box[0] + (box[2] / 2)
                #y2 = box[1] + (box[3] / 2)
                #print(f"MDW: upper left is {x1}, {y1} - lower right {x2}, {y2}")
                #x1 = x1 * 416.0
                #y1 = y1 * 416.0
                #x2 = x2 * 416.0
                #y2 = y2 * 416.0
                #w = x2-x1
                #h = y2-y1
                #print(f"MDW: scaled upper left is {x1}, {y1} - scaled lower right {x2}, {y2}")

                # These are in 'grid units' and need to be mapped back to pixels.
                # Also, remember that the original input images may have been
                # rescaled by the model.
                # XXX - Don't hardcode 416x416 original input size here.
                #x = int(box[0] * (416.0 / grid_cells_x))
                #y = int(box[1] * (416.0 / grid_cells_y))
                #w = int(box[2] * (416.0 / grid_cells_x))
                #h = int(box[3] * (416.0 / grid_cells_y))

                print(
                    f"Batch [{batch_index}] box [{box_index}] conf {boxconf} top class {topclass} "
                    f"(conf {topclass_conf}) = {box}"
                )

    break  # XXX MDW
