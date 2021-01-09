#!/usr/bin/env python

import argparse
import datetime
import json
import os
import random
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

from ldraw.colour import Colour
from ldraw.figure import *
from ldraw.library.colours import *
from ldraw.library.parts.brick import *
from ldraw.pieces import Piece
from ldraw.tools import get_model
from ldraw.writers.povray import POVRayWriter
from PIL import Image, ImageDraw
from progress.bar import Bar
from progress.spinner import Spinner


def get_all_parts() -> List[Any]:
    parts = []
    # Brick parts.
    import ldraw.library.parts.brick as brick

    parts.extend(
        [
            brick.__dict__[p]
            for p in brick.__dict__.keys()
            if re.match(r"^Brick(\d+)X(\d+)$", p)
        ]
    )
    parts.extend(
        [
            brick.__dict__[p]
            for p in brick.__dict__.keys()
            if re.match(r"^Brick(\d+)X(\d+)X(\d+)$", p)
        ]
    )
    return parts


def gen_colors(num_colors: int) -> List[Colour]:
    retval = []
    all_colors = list(ColoursByName)
    for index in range(num_colors):
        colorname = all_colors[index]
        # This one has a special meaning to the LDraw library.
        if colorname != "Main_Colour":
            retval.append(ColoursByName[colorname])
    return retval


POV_HEADER = """
#include "colors.inc"
#include "rad_def.inc"

global_settings {
       max_trace_level 10
       radiosity {
               Rad_Settings(Radiosity_Fast, on, off)
       }
}
"""

# light_source
# { <0, 1900, 0>, rgb <0.95, 0.83, 0.51>
#  fade_distance 1900 fade_power 2
#  area_light x*70, y*70, 20, 20 circular orient adaptive 0 jitter
# }
#
# light_source
# { <4000, 2000, 250>, rgb <1.0, 1.0, 1.0>
#  fade_distance 10000 fade_power 2
#  area_light x*70, y*70, 20, 20 circular orient adaptive 0 jitter
# }


POV_TRAILER = """
light_source
{ <0, 1900, 0>
  color rgb 0.75
  area_light 200, 200, 10, 10
  jitter
}

light_source
{ <4000, 2000, 250>
  color rgb 0.75
  area_light 200, 200, 10, 10
  jitter
}

background { color Black }

camera {
  up <0,1,0>
  right <1,0,0>
  location <-500.000000, 700.000000, -500.000000>
  look_at <500.000000, -200.000000, 500.000000>
  angle 35 
}
"""


def gen_piece(
    part: str,
    color: Colour,
    x_range: Tuple[int, int] = (-500, 2000),
    y_range: Tuple[int, int] = (0, 100),
    z_range: Tuple[int, int] = (-500, 2000),
) -> Piece:
    """Generate a single Piece."""
    x = random.randint(x_range[0], x_range[1])
    y = random.randint(y_range[0], y_range[1])
    z = random.randint(z_range[0], z_range[1])
    xrot = random.randint(0, 360) - 180
    yrot = random.randint(0, 360) - 180
    zrot = random.randint(0, 360) - 180
    rot = Identity().rotate(x, XAxis).rotate(yrot, YAxis).rotate(zrot, ZAxis)
    return Piece(color, Vector(x, y, z), rot, part)


def gen_pile(
    parts: Set[str],
    colors: Set[Colour],
    num_parts: int = 1000,
    x_range: Tuple[int, int] = (-500, 2000),
    y_range: Tuple[int, int] = (0, 100),
    z_range: Tuple[int, int] = (-500, 2000),
) -> List[Piece]:
    """Generate a POVRay scene with `num_parts` random parts in a pile."""
    retval = []
    parts = list(parts)
    colors = list(colors)
    for _ in range(num_parts):
        part = random.choice(parts)
        color = random.choice(colors)
        piece = gen_piece(part, color, x_range, y_range, z_range)
        retval.append(piece)
    return retval


def gen_ldr(ldraw_path, parts):
    """Generate an LDR file from the given list of parts."""
    with open(ldraw_path, "w") as ldr_file:
        for part in parts:
            ldr_file.write(str(part) + "\n")
    # print(f"Wrote LDR file to {ldraw_path}")


def gen_pov(ldraw_path, pov_path):
    """Generate a POVRay file from the given LDR file."""
    model, parts = get_model(ldraw_path)

    with open(pov_path, "w") as pov_file:
        pov_file.write(POV_HEADER + "\n")
        writer = POVRayWriter(parts, pov_file)
        writer.write(model)
        pov_file.write(POV_TRAILER + "\n")
    # print(f"Wrote POV file to {pov_path}")


def run_pov(pov_path, image_path, image_width, image_height):
    """Run POVRay with the given input POV file, output image, and image size."""
    cmd = [
        "povray",
        f"-i{pov_path}",
        f"+W{image_width}",
        f"+H{image_height}",
        "+FN",
        "+A0.3",
        f"-o{image_path}",
    ]
    result = subprocess.run(cmd, capture_output=True)
    result.check_returncode()


def render(pieces: List[Piece], image_path: str, width: int, height: int):
    """Render the given list of pieces to the given image file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ldr") as ldr:
        ldr.close()
        gen_ldr(ldr.name, pieces)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pov") as pov:
        pov.close()
        gen_pov(ldr.name, pov.name)
        run_pov(pov.name, image_path, width, height)
    os.remove(ldr.name)
    os.remove(pov.name)


def get_bounding_box(
    piece: Piece, image_width: int, image_height: int
) -> Optional[Tuple[int, int, int, int]]:
    """Return the bounding box of the given piece in a scene."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".png") as tmpimg:
        tmpimg.close()
        render([piece], tmpimg.name, image_width, image_height)
        img = Image.open(tmpimg.name)
        # This is in format left, upper, right, lower.
        bbox = img.getbbox()
    os.remove(tmpimg.name)
    return bbox


class DatasetImage:
    def __init__(
        self,
        id: int,
        image_fname: str,
        annotations: List[Tuple[str, Tuple[int, int, int, int]]],
    ):
        self._id = id
        self._image_fname = image_fname
        self._annotations = annotations

    @property
    def id(self) -> int:
        return self._id

    @property
    def image_fname(self) -> str:
        return self._image_fname

    @property
    def annotations(self) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        return self._annotations

    def categories(self) -> Set[str]:
        return set([x[0] for x in self._annotations])


def gen_dataset_image(
    index: int,
    total_images: int,
    foreground_parts: Set[Any],
    foreground_colors: Set[Any],
    background_parts: Set[Any],
    background_colors: Set[Any],
    detections_size: int,
    pile_size: int,
    outdir: str,
    image_width: int,
    image_height: int,
) -> DatasetImage:
    """Generate a single image in the dataset."""

    image_fname = os.path.join(outdir, "images", "image_{:05d}.png".format(index))
    annotations = []

    foreground_pieces = []

    foreground_parts = list(foreground_parts)
    foreground_colors = list(foreground_colors)

    with Bar(
        f"[{index}/{total_images}] Generating bounding boxes", max=detections_size
    ) as bar:
        for piece_index in range(detections_size):
            part = random.choice(foreground_parts)
            color = random.choice(foreground_colors)
            piece_name = f"{part}_{color.name}"

            # Render it solo and get its bounding box. Note that this can fail
            # if the image is out of the field of view.
            bbox: Optional[Tuple[int, int, int, int]] = None
            tries = 0
            while bbox is None and tries < 100:
                piece = gen_piece(
                    part,
                    color,
                    x_range=(-250, 1250),
                    y_range=(-200, 0),
                    z_range=(-250, 1250),
                )
                bbox = get_bounding_box(piece, image_width, image_height)
                tries += 1
            if tries == 100:
                print(
                    f"Warning: Unable to generate bbox for {piece_name} after 100 tries."
                )
                continue
            annotations.append((piece_name, bbox))
            foreground_pieces.append(piece)
            bar.next()

    print(f"[{index}/{total_images}] Rendering scene...")
    pile = gen_pile(background_parts, background_colors, pile_size)
    render(pile + foreground_pieces, image_fname, image_width, image_height)
    return DatasetImage(index, image_fname, annotations)


def gen_category_map(all_images: List[DatasetImage]) -> Dict[str, int]:
    """Return a mapping from category name to category ID."""
    category_names = set()
    for image in all_images:
        category_names.update(image.categories())
    return {catname: index for index, catname in enumerate(category_names)}


def gen_category_metadata(all_categories: Dict[str, int]) -> List[Dict[str, Any]]:
    """Return MSCOCO category metadata for the given category map."""
    return [{"id": catid, "name": catname} for catname, catid in all_categories.items()]


def gen_image_metadata(
    images: List[DatasetImage], image_width: int, image_height: int
) -> List[Dict[str, Any]]:
    """Return MSCOCO image metadata for the given images."""
    return [
        {
            "id": img.id,
            "file_name": os.path.basename(img.image_fname),
            "width": image_width,
            "height": image_height,
        }
        for index, img in enumerate(images)
    ]


def gen_annotations(
    all_categories: Dict[str, int], images: List[DatasetImage]
) -> List[Dict[str, Any]]:
    """Generate MSCOCO annotations for the given set of images, using the given category ID map."""
    retval: List[Dict[str, Any]] = []
    index = 0
    for image in images:
        for annotation in image.annotations:
            retval.append(
                {
                    "id": index,
                    "image_id": image.id,
                    "category_id": all_categories[annotation[0]],
                    "bbox": annotation[1],
                }
            )
            index += 1
    return retval


def write_dataset(images: List[DatasetImage], outdir: str, outfile: str):
    with open(os.path.join(outdir, outfile), "w") as outfile:
        for img in images:
            img_fname = os.path.basename(img.image_fname)  # image_00000.png
            fname_base, _ = os.path.splitext(img_fname)    # image_00000
            label_fname = f"{fname_base}.txt")             # image_00000.txt
            outfile.write(f"images/{img_fname} labels/{label_fname}\n")
            with open(os.path.join(outdir, "labels", label_fname), "w") as labelfile:
                for category, bbox in img.annotations:
                    left, upper, right, lower = bbox
                    labelfile.write(
                        f"{category} {left} {upper} {right - left} {lower - upper}\n"
                    )


def gen_dataset(args):
    """Generate the dataset."""

    if os.path.exists(args.outdir):
        if args.overwrite:
            print("Warning: Output path exists, overwriting existing dataset!")
        else:
            raise RuntimeError(
                f"Output path {args.outdir} exists, use --overwrite to overwrite."
            )

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "debug_images"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "labels"), exist_ok=True)

    all_parts = get_all_parts()
    foreground_parts = set(all_parts[0 : args.num_parts])
    background_parts = set(all_parts[0 : args.background_parts])
    background_parts -= foreground_parts

    foreground_colors = set(gen_colors(args.num_colors))
    background_colors = set(gen_colors(args.background_colors))
    background_colors -= foreground_colors

    # Generate images and labels.
    all_images: List[DatasetImage] = []
    for index in range(args.num_images):
        dsimage = gen_dataset_image(
            index,
            args.num_images,
            foreground_parts,
            foreground_colors,
            background_parts,
            background_colors,
            args.detections_size,
            args.pile_size,
            args.outdir,
            args.width,
            args.height,
        )
        all_images.append(dsimage)

        img = Image.open(dsimage.image_fname)
        draw = ImageDraw.Draw(img)
        for category, bbox in dsimage.annotations:
            draw.rectangle(bbox, outline=(0, 255, 0), fill=None, width=1)
            draw.text((bbox[0], bbox[1]), category, (0, 255, 0))
        outfile = os.path.join(
            args.outdir, "debug_images", "image_bboxes_{:05d}.png".format(index)
        )
        img.save(outfile)
        print(f"Saved {outfile}")

    # Generate train/val split.
    assert 0.0 < args.frac_train_images <= 1.0
    split = int(len(all_images) * args.frac_train_images)
    train_images = all_images[0:split]
    val_images = all_images[split:]

    print(f"Writing {len(train_images)}/{len(all_images)} train image annotations...")
    write_dataset(train_images, args.outdir, "train.txt")
    print(f"Writing {len(val_images)}/{len(all_images)} val image annotations...")
    write_dataset(val_images, args.outdir, "val.txt")
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", help="Output directory for generated dataset.", required=True
    )
    parser.add_argument(
        "--num_images",
        help="Number of output images in the dataset.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--width",
        help="Width in pixels of each output image.",
        default=512,
        type=int,
    )
    parser.add_argument(
        "--height",
        help="Height in pixels of each output image.",
        default=512,
        type=int,
    )
    parser.add_argument(
        "--num_parts",
        help="Number of different parts in detection dataset.",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--num_colors",
        help="Number of colors in the detection dataset.",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--background_parts",
        help="Number of different parts in background pile.",
        default=15,
        type=int,
    )
    parser.add_argument(
        "--background_colors",
        help="Number of colors in the background pile.",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--detections_size",
        help="Total number of parts to be detected in each image.",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--pile_size",
        help="Total number of parts in background pile.",
        default=200,
        type=int,
    )
    parser.add_argument(
        "--frac_train_images",
        default=0.7,
        help="Fraction of images in train vs. validation set",
        type=float,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing --outdir if it exists.",
    )

    args = parser.parse_args()

    gen_dataset(args)


if __name__ == "__main__":
    main()
