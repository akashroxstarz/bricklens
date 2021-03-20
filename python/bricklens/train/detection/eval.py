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
from PIL import Image, ImageDraw
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
    "--mscoco_path",
    type=str,
    default=None,
    help="Path to root of MSCOCO dataset. Overrides --dataset_config_path.",
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
    default=2,
    help="number of images to process",
)
parser.add_argument(
    "--use_cuda", type=bool, default=True, help="whether to use cuda if available"
)
parser.add_argument(
    "--obj_threshold",
    type=float,
    default=0.6,
    help="Objectness threshold for each bounding box",
)
args = parser.parse_args()
print(args)

# Check for CUDA.
cuda = torch.cuda.is_available() and args.use_cuda
print(f"Using CUDA: {cuda}")
device = "cuda:0" if cuda else "cpu"

# Get dataset configuration.
if args.mscoco_path is not None:
    coco_train_path = os.path.join(args.mscoco_path, "train2014")
    coco_train_annotations_path = os.path.join(
        args.mscoco_path, "annotations", "instances_train2014.json"
    )
    coco_val_path = os.path.join(args.mscoco_path, "val2014")
    coco_val_annotations_path = os.path.join(
        args.mscoco_path, "annotations", "instances_val2014.json"
    )
    dset = dataset.CocoDataset(coco_val_path, coco_val_annotations_path)
    val_dataloader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
        num_workers=args.n_cpu,
    )

else:
    # Get dataset configuration.
    with open(args.dataset_config_path, "r") as stream:
        data_config = yaml.safe_load(stream)
    train_path = data_config["train"]
    val_path = data_config["val"]
    classfile_path = data_config["classes"]

    # Create dataloader.
    dset = dataset.ListDataset(val_path, classfile_path)
    val_dataloader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
        num_workers=args.n_cpu,
    )


# Get model parameters.
with open(args.model_config_path, "r") as stream:
    model_config = yaml.safe_load(stream)

# Create model.
model = Darknet(model_config, training=False)
model.load_weights(args.weights_path)
print(f"Loaded weights from {args.weights_path}")
if cuda:
    model = model.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


for index, (fnames, imgs, targets) in enumerate(val_dataloader):
    assert len(fnames) == len(imgs) == len(targets) == 1

    if index > args.num_images:
        break

    input_image = np.array(Image.open(fnames[0]))

    assert imgs.size(2) == imgs.size(3)  # Expect square image from the dataloader.
    img_dim = imgs.size(2)

    print("")
    print("")
    print(f"Input index [{index}]: {fnames[0]}")
    print(f"  Image has size {imgs[0].size()}")
    print(f"  Target has size {targets[0].size()}")

    imgs = Variable(imgs.type(Tensor))
    targets = Variable(targets.type(Tensor), requires_grad=False)
    with torch.no_grad():
        detections, loss = model(imgs, targets)
        if detections is None:
            print(f"WARNING: No detections for index [{index}]")
            continue
        print(f"Got raw detections: {detections.shape}")
        print(f"Got raw loss: {loss}")
        print(f"Model loss: {model.losses}")
        detections = utils.non_max_suppression(
            detections, num_classes=40, conf_thres=args.obj_threshold, nms_thres=0.4
        )[0]
        #print(f"Post-NMS detections: {detections.shape}")

    if detections is None:
        print(f"WARNING: No detections for index [{index}]")
        continue

    # Rescale detection bboxes to size of original input image.
    detections = utils.rescale_boxes(detections, img_dim, input_image.shape[:2])
    img = Image.open(fnames[0])
    draw = ImageDraw.Draw(img)
    for detnum, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
        topclass = dset.classindex_to_name(cls_pred)
        print(
            f"Detection [{detnum}] = bbox conf {conf:.4f} class {topclass} cls_conf {cls_conf:.4f} = ({int(x1)},{int(y1)})..({int(x2)},{int(y2)})"
        )
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0))
        draw.text((x1, y1-10), topclass, fill=(255,0,0))
    img.save("mdw_" + os.path.basename(fnames[0]))


