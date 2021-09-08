#!/usr/bin/env python

import argparse
import datetime
import itertools
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
from rich.console import Console
import rich.progress

import bricklens.render.blender_utils as blender_utils

TEMPLATE_FILE_WITH_BACKGROUND = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "emptyscene.blend")
)
TEMPLATE_FILE_NO_BACKGROUND = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "emptybackground.blend")
)

console = Console()


def get_all_parts() -> List[Any]:
    parts = []
    # Brick parts.
    import ldraw.library.parts.brick as brick

    all_bricks = list(brick.__dict__.keys())
    all_bricks.sort()

    parts.extend(
        [
            brick.__dict__[p]
            for p in all_bricks
            # For now, limit to smaller bricks.
            if re.match(r"^Brick[12345678]X[12345678]$", p)
        ]
    )
    #    parts.extend(
    #        [
    #            brick.__dict__[p]
    #            for p in brick.__dict__.keys()
    #            if re.match(r"^Brick(\d+)X(\d+)X(\d+)$", p)
    #        ]
    #    )
    parts.sort()
    return parts


def get_all_colors() -> List[Colour]:
    retval = []
    # The first 16 are the primary colors.
    all_colors = list(ColoursByName)
    all_colors.sort()
    all_colors = all_colors[0:15]
    for index in range(len(all_colors)):
        colorname = all_colors[index]
        # These have a special meaning to the LDraw library.
        if colorname != "Main_Colour" and colorname != "Edge_Colour":
            retval.append(ColoursByName[colorname])
    return retval


def gen_piece(
    part: str,
    color: Colour,
    x_range: Tuple[int, int] = (-500, 500),
    y_range: Tuple[int, int] = (0, 100),
    z_range: Tuple[int, int] = (-500, 500),
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
    x_range: Tuple[int, int] = (-500, 500),
    y_range: Tuple[int, int] = (0, 100),
    z_range: Tuple[int, int] = (-500, 500),
) -> List[Piece]:
    """Generate a set of Pieces with `num_parts` random parts in a pile."""
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


def gen_pov(ldraw_path, pov_path):
    """Generate a POVRay file from the given LDR file."""
    model, parts = get_model(ldraw_path)

    with open(pov_path, "w") as pov_file:
        pov_file.write(POV_HEADER + "\n")
        writer = POVRayWriter(parts, pov_file)
        writer.write(model)
        pov_file.write(POV_TRAILER + "\n")


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


def render(
    pieces: List[Piece],
    image_path: str,
    ldraw_library_path: str,
    width: int,
    height: int,
    single_piece: bool = False,
):
    """Render the given list of pieces to the given image file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ldr") as ldr:
        ldr.close()
        gen_ldr(ldr.name, pieces)
    if single_piece:
        template_file = TEMPLATE_FILE_NO_BACKGROUND
    else:
        template_file = TEMPLATE_FILE_WITH_BACKGROUND

    blender_utils.render_ldr(
        ldr_file=ldr.name,
        output_file=image_path,
        template_file=template_file,
        ldraw_library_path=ldraw_library_path,
    )

    os.remove(ldr.name)


def get_bounding_box(
    piece: Piece, image_width: int, image_height: int, ldraw_library_path: str
) -> Optional[Tuple[int, int, int, int]]:
    """Return the bounding box of the given piece in a scene."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".png") as tmpimg:
        tmpimg.close()
        render(
            [piece],
            tmpimg.name,
            ldraw_library_path,
            image_width,
            image_height,
            single_piece=True,
        )
        img = Image.open(tmpimg.name)
        # This is in format left, upper, right, lower.
        bbox = img.getbbox()
    os.remove(tmpimg.name)
    return bbox


class DatasetImage:
    def __init__(
        self,
        id: int,
        image_width: int,
        image_height: int,
        image_fname: str,
        annotations: List[Tuple[str, Tuple[int, int, int, int]]],
    ):
        self._id = id
        self._width = image_width
        self._height = image_height
        self._image_fname = image_fname
        self._annotations = annotations

    @property
    def id(self) -> int:
        return self._id

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def image_fname(self) -> str:
        return self._image_fname

    @property
    def annotations(self) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        return self._annotations

    def categories(self) -> Set[str]:
        return set([x[0] for x in self._annotations])


def gen_classes(parts: Set[Any], colors: Set[Any]) -> List[str]:
    """Generate list of classes in the dataset."""
    parts = list(parts)
    colors = list(colors)
    return [f"{part}_{color.name}" for part, color in itertools.product(parts, colors)]


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
    classes: List[str],
    ldraw_library_path: str,
) -> DatasetImage:
    """Generate a single image in the dataset."""

    image_fname = os.path.join(outdir, "images", "image_{:05d}.png".format(index))
    console.log(f"Generating {image_fname}...")
    annotations = []

    foreground_pieces = []

    foreground_parts = list(foreground_parts)
    foreground_colors = list(foreground_colors)

    progress = rich.progress.Progress(
        rich.progress.SpinnerColumn(),
        "[progress.description]{task.description}",
        rich.progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        rich.progress.TimeElapsedColumn(),
    )

    with progress:
        task1 = progress.add_task("[green]Generating parts...", total=detections_size)

        for piece_index in range(detections_size):
            progress.update(task1, advance=1)

            part = random.choice(foreground_parts)
            color = random.choice(foreground_colors)
            piece_name = f"{part}_{color.name}"
            assert (
                piece_name in classes
            ), f"Output classe list does not contain piece name {piece_name}: {classes}"
            classid = classes.index(piece_name)

            # Render it solo and get its bounding box. Note that this can fail
            # if the image is out of the field of view.
            bbox: Optional[Tuple[int, int, int, int]] = None
            tries = 0
            while bbox is None and tries < 100:
                piece = gen_piece(
                    part,
                    color,
                    # Try to avoid the target piece going outside the camera view.
                    x_range=(-300, 300),
                    # Want target pieces to generally be above pile.
                    y_range=(-100, 0),
                    z_range=(-300, 300),
                )
                bbox = get_bounding_box(
                    piece, image_width, image_height, ldraw_library_path
                )
                tries += 1
            if tries == 100:
                console.log(
                    f"[bold red]Warning: Unable to generate bbox for {piece_name} after 100 tries."
                )
                continue
            annotations.append((str(classid), bbox))
            foreground_pieces.append(piece)

    with console.status("[green]Rendering..."):
        pile = gen_pile(background_parts, background_colors, pile_size)
        render(
            pile + foreground_pieces,
            image_fname,
            ldraw_library_path,
            image_width,
            image_height,
        )

    console.log(f"[red]Done![/red] Wrote {image_fname}")
    return DatasetImage(index, image_width, image_height, image_fname, annotations)


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
            fname_base, _ = os.path.splitext(img_fname)  # image_00000
            label_fname = f"{fname_base}.txt"  # image_00000.txt
            outfile.write(f"images/{img_fname} labels/{label_fname}\n")
            with open(os.path.join(outdir, "labels", label_fname), "w") as labelfile:
                for category, bbox in img.annotations:
                    left, upper, right, lower = bbox

                    # Normalize bbox locations as required by Yolov3 code.
                    left = (left * 1.0) / img.width
                    upper = (upper * 1.0) / img.height
                    right = (right * 1.0) / img.width
                    lower = (lower * 1.0) / img.height

                    # Yolov3 expects bbox coordinates in the form:
                    #   center_x center_y box_width box_height
                    box_width = right - left
                    box_height = lower - upper
                    center_x = left + (box_width / 2.0)
                    center_y = upper + (box_height / 2.0)
                    labelfile.write(
                        f"{category} {center_x} {center_y} {box_width} {box_height}\n"
                    )


def gen_dataset(args):
    """Generate the dataset."""
    console.rule("[bold red]Bricklens dataset generator")
    console.log(f"Writing dataset to [red]{args.outdir}[/red]")
    rng = random.Random(args.seed)  # So we get consistent selection of parts+colors.

    if os.path.exists(args.outdir):
        if args.overwrite:
            console.log(
                "[bold red]Warning:[/bold red] Output path exists, overwriting existing dataset!"
            )
        else:
            raise RuntimeError(
                f"Output path {args.outdir} exists, use --overwrite to overwrite."
            )

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "debug_images"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "labels"), exist_ok=True)

    # The below is a little convoluted to ensure we get the same set of parts and
    # colors for the same value of args.seed.
    all_parts = get_all_parts()
    foreground_parts = rng.sample(all_parts, min(len(all_parts), args.num_parts))
    all_parts = list(sorted(set(all_parts) - set(foreground_parts)))
    background_parts = random.sample(
        all_parts, min(len(all_parts), args.background_parts)
    )
    console.log(
        f"Using [blue]{len(foreground_parts)}[/blue] foreground and "
        f"[blue]{len(background_parts)}[/blue] background parts."
    )

    all_colors = get_all_colors()
    foreground_colors = rng.sample(all_colors, min(len(all_colors), args.num_colors))
    all_colors = list(set(all_colors) - set(foreground_colors))
    background_colors = random.sample(
        all_colors, min(len(all_colors), args.background_colors)
    )
    console.log(
        f"Using [blue]{len(foreground_colors)}[/blue] foreground and "
        f"[blue]{len(background_colors)}[/blue] background colors."
    )

    # Generate class mapping file.
    classes = gen_classes(foreground_parts, foreground_colors)
    with open(os.path.join(args.outdir, "classes.txt"), "w") as classfile:
        for index, classname in enumerate(classes):
            classfile.write(f"{index} {classname}\n")
    console.log(f"Creating [blue]{len(classes)}[/blue] classes.")

    if args.skip_rendering:
        console.log("[red]Skipping image rendering.")
        return

    # Generate images and labels.
    all_images: List[DatasetImage] = []

    for index in range(args.num_images):
        console.rule(f"[green]Image {index}/{args.num_images-1}")

        detections_size = int(random.uniform(args.detections_min, args.detections_max+1))
        pile_size = int(random.uniform(args.pile_min, args.pile_max+1))
        dsimage = gen_dataset_image(
            index,
            args.num_images,
            foreground_parts,
            foreground_colors,
            background_parts,
            background_colors,
            detections_size,
            pile_size,
            args.outdir,
            args.width,
            args.height,
            classes,
            args.ldraw_library_path,
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

    # Generate train/val split.
    assert 0.0 < args.frac_train_images <= 1.0
    split = int(len(all_images) * args.frac_train_images)
    train_images = all_images[0:split]
    val_images = all_images[split:]

    console.log(
        f"Writing [blue]{len(train_images)}[/blue]/[blue]{len(all_images)}[/blue] train image "
        "annotations..."
    )
    write_dataset(train_images, args.outdir, "train.txt")
    console.log(
        f"Writing [blue]{len(val_images)}[/blue]/[blue]{len(all_images)}[/blue] val image "
        "annotations..."
    )
    write_dataset(val_images, args.outdir, "val.txt")
    console.log("[bold green]Done.")
    console.log(f"Your dataset is in: [red]{args.outdir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", help="Output directory for generated dataset.", required=True
    )
    parser.add_argument(
        "--seed",
        help="Random seed used to generate piece and color lists.",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--skip_rendering",
        action="store_true",
        default=False,
        help="Skip rendering images and generating labels. If set, only write class file.",
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
        default=4,
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
        default=4,
        type=int,
    )
    parser.add_argument(
        "--background_colors",
        help="Number of colors in the background pile.",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--detections_min",
        help="Min number of parts to be detected in each image.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--detections_max",
        help="Max number of parts to be detected in each image.",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--pile_min",
        help="Min number of parts in background pile.",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--pile_max",
        help="Max number of parts in background pile.",
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
    parser.add_argument(
        "--ldraw_library_path",
        help="Path to LDRAW library",
        required=True,
    )

    args = parser.parse_args()

    gen_dataset(args)


if __name__ == "__main__":
    main()
