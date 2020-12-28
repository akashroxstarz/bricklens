# Adapted from:
# https://github.com/cfotache/pytorch_custom_yolo_training

import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from skimage.transform import resize
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ListDataset(Dataset):
    """Dataset loader for a simple dataset format. The dataset is described with a text file,
    consisting of a list of full image pathnames, one per row.
    """

    def __init__(self, list_path: str, image_shape: Tuple[int, int] = (416, 416)):
        with open(list_path, "r") as infile:
            list_rows = infile.readlines()

        self._image_filenames: List[str] = []
        self._label_filenames: List[str] = []

        for row in list_rows:
            assert len(row.split()) == 2
            image_filename, label_filename = row.split()
            assert os.path.exists(image_filename)
            assert os.path.exists(label_filename)
            self._image_filenames.append(image_filename)
            self._label_filenames.append(label_filename)
        self._image_shape = image_shape
        self.max_objects = 1

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, List[torch.Tensor]]:
        """PyTorch Dataset getitem method."""

        # ---------
        #  Image
        # ---------

        img_path = self._image_filenames[index % len(self._image_filenames)]
        img = np.array(Image.open(img_path))
        assert (
            len(img.shape) == 3
        ), f"{img_path} has shape {img.shape}, expect three channels."

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (
            ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        )
        # Add padding
        input_img = np.pad(img, pad, "constant", constant_values=128) / 255.0
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode="reflect")
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        # ---------
        #  Labels
        # ---------

        label_path = _self.label_filenames[index % len(self.img_files)]
        labels = np.loadtxt(label_path).reshape(-1, 5)
        if len(labels) == 0:
            # Empty file, no labels.
            return img_path, input_img, torch.from_numpy(labels)

        # Format of rows in the labels file is expected to be:
        # class x y width height

        # Extract coordinates for unpadded + unscaled image
        x1 = w * (labels[:, 1] - labels[:, 3] / 2)
        y1 = h * (labels[:, 2] - labels[:, 4] / 2)
        x2 = w * (labels[:, 1] + labels[:, 3] / 2)
        y2 = h * (labels[:, 2] + labels[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates
        labels[:, 1] = ((x1 + x2) / 2) / padded_w
        labels[:, 2] = ((y1 + y2) / 2) / padded_h
        labels[:, 3] *= w / padded_w
        labels[:, 4] *= h / padded_h

        # XXX MDW - Unsure of need for no more than "max_objects" labels per image.
        # May depend on how the loss function interprets the returned tensor.

        # Fill matrix
        # filled_labels = np.zeros((self.max_objects, 5))
        # if labels is not None:
        #    filled_labels[range(len(labels))[: self.max_objects]] = labels[
        #        : self.max_objects
        #    ]
        # filled_labels = torch.from_numpy(filled_labels)
        filled_labels = torch.from_numpy(labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self._image_files)
