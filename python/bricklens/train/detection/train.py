#!/usr/bin/env python3

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
import wandb
import yaml
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataset
from model import Darknet

# Get command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
parser.add_argument(
    "--batch_size", type=int, default=16, help="size of each image batch"
)
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

# Check for CUDA.
cuda = torch.cuda.is_available() and args.use_cuda
print(f"Using CUDA: {cuda}")
device = "cuda:0" if cuda else "cpu"

# Create checkpoint directory.
os.makedirs(args.checkpoint_dir, exist_ok=True)

# Get dataset configuration.
with open(args.dataset_config_path, "r") as stream:
    data_config = yaml.safe_load(stream)
train_path = data_config["train"]

# Get model parameters.
with open(args.model_config_path, "r") as stream:
    model_config = yaml.safe_load(stream)

# Create model.
model = Darknet(model_config)
print(f"MDW: model is:\n{model}")
if args.weights_path is not None and os.path.exists(args.weights_path):
    model.load_weights(args.weights_path)
    # model.apply(weights_init_normal)
if cuda:
    model = model.cuda()
model.train()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Create dataloader.
dataloader = torch.utils.data.DataLoader(
    dataset.ListDataset(train_path),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.n_cpu,
)

# Create optimizer.
initial_lr = float(model_config["hyperparams"]["initial_lr"])
gamma = float(model_config["hyperparams"]["gamma"])
# momentum = float(model_config["momentum"])
# decay = float(model_config["decay"])
# burn_in = int(model_config["burn_in"])
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr
)
scheduler = StepLR(optimizer, step_size=((args.epochs * 0.9) // 3), gamma=gamma)

# Initialize WandB.
WORKLOAD = "bricklens"
wandb.init(project=WORKLOAD)
config = wandb.config
config.max_epochs = args.epochs
config.initial_lr = initial_lr
config.gamma = gamma
wandb.watch(model)

# Do the training loop.
for epoch in range(args.epochs):
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

        # Log to WandB.
        for index, param_group in enumerate(optimizer.param_groups):
            ldict[f"lr_{index}"] = param_group["lr"]
        wandb.log(ldict)
        wandb.log({"loss_x": model.losses["x"]})
        wandb.log({"loss_y": model.losses["y"]})
        wandb.log({"loss_w": model.losses["w"]})
        wandb.log({"loss_h": model.losses["h"]})
        wandb.log({"loss_conf": model.losses["conf"]})
        wandb.log({"loss_cls": model.losses["cls"]})
        wandb.log({"loss": loss.item()})
        wandb.log({"precision": model.losses["precision"]})
        wandb.log({"recall": model.losses["recall"]})

    if epoch % args.checkpoint_interval == 0:
        model.save_weights(os.path.join(args.checkpoint_dir, "f{epoch}.weights"))
