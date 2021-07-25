# Adapted from:
# https://github.com/cfotache/pytorch_custom_yolo_training

import os
from typing import List, Tuple

import numpy as np
import torch
import torchvision.datasets as dset
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

        self._classindices: Dict[str, float] = {}
        self._classnames: Dict[int, str] = {}
        with open(classfile_path, "r") as infile:
            for row in infile:
                classnum, classname = row.split()
                self._classindices[classname] = float(classnum)
                self._classnames[int(classnum)] = classname

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

    def classname_to_index(self, class_name: str) -> float:
        """Convert class name to an index value."""
        return self._classindices[class_name]

    def classindex_to_name(self, class_index: float) -> str:
        """Convert class index to name."""
        return self._classnames.get(int(class_index), f"<unknown class {class_index}>")

    @property
    def num_classes(self) -> int:
        return len(self._classindices)

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
            label_path,
            converters={0: lambda s: self.classname_to_index(s.decode("utf-8"))},
        ).reshape(-1, 5)

        if len(labels) == 0:
            # Empty file, no labels.
            return input_img, torch.from_numpy(labels)

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


class CocoDataset(Dataset):
    """Dataset loader for the MSCOCO dataset format."""

    def __init__(
        self,
        root_path: str,
        ann_file_path: str,
        image_shape: Tuple[int, int] = (416, 416),
        max_objects_per_image: int = 20,
    ):
        self._root_path = root_path
        self._coco = dset.CocoDetection(root=root_path, annFile=ann_file_path)

        self._image_shape = image_shape
        self._max_objects_per_image = max_objects_per_image

    @property
    def num_classes(self) -> int:
        return len(self._coco.coco.dataset["categories"])

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, List[torch.Tensor]]:
        """PyTorch Dataset getitem method."""

        img, raw_labels = self._coco[index]
        print(f"MDW: img is: {img}")
        # Bit of a hack here to get the filename.
        img_id = self._coco.ids[index]
        fname = self._coco.coco.loadImgs(img_id)[0]["file_name"]
        fname = os.path.join(self._root_path, fname)
        print(f"MDW: fname is: {fname}")

        # ---------
        #  Image
        # ---------

        img = np.array(img)
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

        # Map from MSCOCO annotation format to what we're using internally.
        labels = []
        for label in raw_labels:
            catid = label["category_id"]
            if catid >= self.num_classes:
                # PJReddie's pretrained YOLO3 for MSCOCO mistakenly uses only 80 classes.
                # We ignore GT labels outside of this range.
                continue
            catname = self.classindex_to_name(label["category_id"])
            print(
                f"MDW: GT cat {label['category_id']} name {catname} bbox {label['bbox']}"
            )
            # MSCOCO categories are 1-indexed, so we subtract one from the ID.
            labels.append(
                [
                    label["category_id"] - 1,
                    label["bbox"][0],
                    label["bbox"][1],
                    label["bbox"][2],
                    label["bbox"][3],
                ]
            )

        labels = np.array(labels).reshape(-1, 5)

        if len(labels) == 0:
            # Empty file, no labels.
            return fname, input_img, torch.from_numpy(labels)

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
        return fname, input_img, filled_labels

    def __len__(self):
        return len(self._coco)

    def classname_to_index(self, class_name: str) -> float:
        """Convert class name to an index value."""
        return self._coco.coco.getCatIds(catNms=[class_name])

    def classindex_to_name(self, class_index: float) -> str:
        """Convert class index to name."""
        cats = [
            cat
            for cat in self._coco.coco.dataset["categories"]
            if cat["id"] == class_index - 1
        ]
        if len(cats) == 0:
            return f"<unknown class {class_index}>"
        else:
            return " ".join([cat["name"] for cat in cats])
