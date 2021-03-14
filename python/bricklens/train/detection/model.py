# Adapted from:
# https://github.com/cfotache/pytorch_custom_yolo_training

from __future__ import division

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from utils import build_targets


def create_modules(
    module_config: Dict[str, Any], training: bool
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], nn.ModuleList]:
    """
    Constructs module list of layer blocks from module configuration in module_config.
    Returns:
      (1) Dict of yperparameters.
      (2) List of dicts of module definitions.
      (3) nn.ModuleList.
    """
    hyperparams = module_config["hyperparams"]
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    module_defs = module_config["modules"]

    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def.get("batch_normalize", "0"))
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            upsample = nn.Upsample(
                scale_factor=int(module_def["stride"]), mode="nearest"
            )
            modules.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"]]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module("route_%d" % i, EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"]]
            # Extract anchors
            anchors = module_def["anchors"]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height, training)
            modules.add_module("yolo_%d" % i, yolo_layer)
        else:
            raise ValueError(f"Unrecognized module type: {module_def}")

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_defs, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """YOLO detection layer."""

    def __init__(self, anchors, num_classes, img_dim, training=True):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1
        self._training = training

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, targets=None):
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG

        # print(
        #    f"MDW: YoloLayer.forward: x.size is {x.size()}, nA {nA} nB {nB} nG {nG} stride {stride}"
        # )

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        BoolTensor = torch.cuda.BoolTensor if x.is_cuda else torch.BoolTensor

        prediction = (
            x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()
        )

        # print(f"MDW: prediction.size() is {prediction.size()}")

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        # grid_x and grid_y are filled with 0, 1, ... nG
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = (
            torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        )

        scaled_anchors = FloatTensor(
            [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        )

        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        if self._training:
            assert targets is not None

        # Calculate loss if we were provided targets
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = 0
            if nProposals > 0:
                precision = float(nCorrect / nProposals)

            # Handle masks
            mask = Variable(mask.type(BoolTensor))
            conf_mask = Variable(conf_mask.type(BoolTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask ^ mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(
                pred_conf[conf_mask_false], tconf[conf_mask_false]
            ) + self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])

            print(f"MDW: tcls is: {tcls}")
            print(f"MDW: tcls[mask] is: {tcls[mask]}")

            loss_cls = (1 / nB) * self.ce_loss(
                pred_cls[mask], torch.argmax(tcls[mask], 1)
            )
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # Training
        if self._training:

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            # XXX MDW Hacking
            # return (pred_boxes, pred_conf, pred_cls)

            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )

            # foo = torch.split(output, [4, 1, 40], -1)
            # print(f"MDW: split is: {len(foo)}, {foo[0].size()} {foo[1].size()} {foo[2].size()}")
            # assert torch.all(torch.eq(foo[0].view(nB, 3, nG, nG, 4) / stride, pred_boxes))
            # assert torch.all(torch.eq(foo[1].view(nB, 3, nG, nG), pred_conf))
            # assert torch.all(torch.eq(foo[2].view(nB, 3, nG, nG, self.num_classes), pred_cls))
            return output


class Darknet(nn.Module):
    """YOLOv3 object detection model."""

    def __init__(
        self, module_config: Dict[str, Any], img_size: int = 416, training=True
    ):
        super(Darknet, self).__init__()
        self._hyperparams, self._module_defs, self._module_list = create_modules(
            module_config, training
        )
        self._img_size = img_size
        self._training = training
        self.seen = 0
        self._header_info = np.array([0, 0, 0, self.seen, 0])
        self._loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    def forward(self, x: torch.Tensor, targets: Optional[List[torch.Tensor]] = None):
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def, module) in enumerate(
            zip(self._module_defs, self._module_list)
        ):
            # print(f"MDW: Layer [{i}]: {module_def}")
            # print(f"[{i}] Pre-layer CUDA memory usage:\n" + torch.cuda.memory_summary())
            # print(f"MDW: Layer [{i}] input size: {x.size()}")

            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"]]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if self._training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self._loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module[0](x, targets)
                output.append(x)
            # print(f"MDW: Layer [{i}] output size: {x.size()}")
            layer_outputs.append(x)

        self.losses["recall"] /= 3
        self.losses["precision"] /= 3
        # print(f"MDW: output is {output}")
        if self._training:
            return sum(output)
        else:
            return torch.cat(output, 1)

    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as fp:
            header = np.fromfile(
                fp, dtype=np.int32, count=5
            )  # First five are header values

            # Needed to write header when saving weights
            self._header_info = header

            self.seen = header[3]
            weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights

        ptr = 0
        for i, (module_def, module) in enumerate(
            zip(self._module_defs, self._module_list)
        ):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if "batch_normalize" in module_def:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.bias
                    )
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.weight
                    )
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.running_mean
                    )
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.running_var
                    )
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        conv_layer.bias
                    )
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(
                    conv_layer.weight
                )
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_weights(self, path, cutoff=-1):
        """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        with open(path, "wb") as fp:
            self._header_info[3] = self.seen
            self._header_info.tofile(fp)

            # Iterate through layers
            for i, (module_def, module) in enumerate(
                zip(self._module_defs[:cutoff], self._module_list[:cutoff])
            ):
                if module_def["type"] == "convolutional":
                    conv_layer = module[0]
                    # If batch norm, load bn first
                    if "batch_normalize" in module_def:
                        bn_layer = module[1]
                        bn_layer.bias.data.cpu().numpy().tofile(fp)
                        bn_layer.weight.data.cpu().numpy().tofile(fp)
                        bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                        bn_layer.running_var.data.cpu().numpy().tofile(fp)
                    # Load conv bias
                    else:
                        conv_layer.bias.data.cpu().numpy().tofile(fp)
                    # Load conv weights
                    conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
