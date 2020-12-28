# Adapted from:
# https://github.com/cfotache/pytorch_custom_yolo_training

from __future__ import division

import argparse
import datetime
import os
import sys
import time

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *

import dataset
import model

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
parser.add_argument(
    "--batch_size", type=int, default=16, help="size of each image batch"
)
parser.add_argument(
    "--model_config_path",
    type=str,
    default="config/yolov3.yml",
    help="Path to model config file",
)
parser.add_argument(
    "--dataset_config_path",
    type=str,
    default="config/dataset.yml",
    help="Path to dataset config file",
)
parser.add_argument(
    "--weights_path",
    type=str,
    default="config/yolov3.weights",
    help="path to weights file",
)
parser.add_argument(
    "--conf_thres", type=float, default=0.8, help="object confidence threshold"
)
parser.add_argument(
    "--nms_thres",
    type=float,
    default=0.4,
    help="IoU thresshold for non-maximum suppression",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=0,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--img_size", type=int, default=416, help="size of each image dimension"
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=1,
    help="interval between saving model weights",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="checkpoints",
    help="directory where model checkpoints are saved",
)
parser.add_argument(
    "--use_cuda", type=bool, default=True, help="whether to use cuda if available"
)
args = parser.parse_args()
print(args)

cuda = torch.cuda.is_available() and args.use_cuda
print(f"Using CUDA: {cuda}")

os.makedirs("checkpoints", exist_ok=True)

# Get dataset configuration.
with open(args.dataset_config_path, "r") as stream:
    data_config = yaml.safe_load(stream)
train_path = data_config["train"]

# Get model parameters.
with open(args.model_config_path, "r") as stream:
    model_config = yaml.safe_load(stream)

learning_rate = float(model_config["learning_rate"])
momentum = float(model_config["momentum"])
decay = float(model_config["decay"])
burn_in = int(model_config["burn_in"])

# Initiate model
model = Darknet(model_config)
if args.weights_path is not None:
    model.load_weights(args.weights_path)
    # model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.n_cpu,
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(opt.epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                args.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)

    if epoch % args.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (args.checkpoint_dir, epoch))


