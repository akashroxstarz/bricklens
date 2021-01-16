# Adapted from:
# https://github.com/cfotache/pytorch_custom_yolo_training

import os
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset


class ListDataset(Dataset):
    """Dataset loader for a simple dataset format. The dataset is described with a text file,
    consisting of a list of full image pathnames, one per row.
    """

    def __init__(
        self,
        list_path: str,
        classfile_path: str,
        image_shape: Tuple[int, int] = (416, 416),
        max_objects_per_image: int = 20,
    ):
        basedir = os.path.dirname(list_path)

        with open(list_path, "r") as infile:
            list_rows = infile.readlines()

        self._classes: Dict[str, float] = {}
        with open(classfile_path, "r") as infile:
            for row in infile:
                classnum, classname = row.split()
                self._classes[classname] = float(classnum)

        self._image_filenames: List[str] = []
        self._label_filenames: List[str] = []

        for row in list_rows:
            assert len(row.split()) == 2
            image_filename, label_filename = row.split()
            image_filename = os.path.join(basedir, image_filename)
            label_filename = os.path.join(basedir, label_filename)
            assert os.path.exists(image_filename), f"{image_filename} not found"
            assert os.path.exists(label_filename), f"{label_filename} not found"
            self._image_filenames.append(image_filename)
            self._label_filenames.append(label_filename)
        self._image_shape = image_shape
        self._max_objects_per_image = max_objects_per_image

    def _convert_class(self, class_name: str) -> float:
        """Convert class name to a float value."""
        return self._classes[class_name]

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
        input_img = resize(input_img, (*self._image_shape, 3), mode="reflect")
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        # ---------
        #  Labels
        # ---------

        # Format of rows in the labels file is expected to be:
        # class x y width height

        label_path = self._label_filenames[index % len(self._image_filenames)]
        labels = np.loadtxt(
            label_path, converters={0: lambda s: self._convert_class(s.decode("utf-8"))}
        ).reshape(-1, 5)

        if len(labels) == 0:
            # Empty file, no labels.
            return img_path, input_img, torch.from_numpy(labels)

        # Extract top-left and bottom-right coords for unpadded + unscaled image
        x1 = labels[:, 1]
        y1 = labels[:, 2]
        x2 = labels[:, 1] + labels[:, 3]
        y2 = labels[:, 2] + labels[:, 4]

        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]

        # Calculate center point and scaled width/height from (0.0, 1.0)
        labels[:, 1] = ((x1 + x2) / 2) / padded_w
        labels[:, 2] = ((y1 + y2) / 2) / padded_h
        labels[:, 3] /= padded_w
        labels[:, 4] /= padded_h

        # Fill matrix
        filled_labels = np.zeros((self._max_objects_per_image, 5))
        if labels is not None:
            filled_labels[range(len(labels))[: self._max_objects_per_image]] = labels[
                : self._max_objects_per_image
            ]
        filled_labels = torch.from_numpy(filled_labels)
        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self._image_filenames)
