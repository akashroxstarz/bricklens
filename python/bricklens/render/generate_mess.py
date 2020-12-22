#!/usr/bin/env python

import argparse
import os
import random
import subprocess
import tempfile
from typing import Any, List, Tuple

from ldraw.figure import *
from ldraw.library.colours import *
from ldraw.library.parts.brick import *
from ldraw.pieces import Piece
from ldraw.tools import get_model
from ldraw.writers.povray import POVRayWriter


PARTS = [
    Brick1X1,
    Brick1X2,
    Brick1X3,
    Brick1X6,
    Brick2X2,
    Brick2X3,
    Brick2X4,
]


def gen_colors(num_colors: int) -> List[Any]:
    retval = []
    all_colors = list(ColoursByName)
    for index in range(num_colors):
        colorname = all_colors[index]
        retval.append(ColoursByName[colorname])
    return retval


# { <2000, 1700, 0>, <1,.8,.4>

# light_source
# { <2000, 1700, 0>, <1,1,1>
#  fade_distance 1000 fade_power 2
#  area_light x*70, y*70, 20, 20 circular orient adaptive 0 jitter
# }


POV_TRAILER = """
light_source
{ <0, 1900, 0>, 1.0
  fade_distance 1900 fade_power 2
  area_light x*70, y*70, 20, 20 circular orient adaptive 0 jitter
}

background { color Gray }

camera {
  location <-500.000000, 700.000000, -500.000000>
  look_at <500.000000, -200.000000, 500.000000>
  angle 40
}
"""


def gen_parts(
    parts: List[Any],
    colors: List[Any],
    num_parts: int = 1000,
    x_range: Tuple[int, int] = (-500, 2000),
    y_range: Tuple[int, int] = (0, 100),
    z_range: Tuple[int, int] = (-500, 2000),
) -> List[Any]:
    """Generate a POVRay scene with `num_parts` random parts in a pile."""
    retval = []
    for _ in range(num_parts):
        part = random.choice(parts)
        color = random.choice(colors)

        x = random.randint(x_range[0], x_range[1])
        y = random.randint(y_range[0], y_range[1])
        z = random.randint(z_range[0], z_range[1])

        xrot = random.randint(0, 360) - 180
        yrot = random.randint(0, 360) - 180
        zrot = random.randint(0, 360) - 180
        rot = Identity().rotate(x, XAxis).rotate(yrot, YAxis).rotate(zrot, ZAxis)

        retval.append(Piece(color, Vector(x, y, z), rot, part))
    return retval


def gen_pov(ldraw_path, pov_path):
    model, parts = get_model(ldraw_path)

    with open(pov_path, "w") as pov_file:
        pov_file.write('#include "colors.inc"\n\n')
        writer = POVRayWriter(parts, pov_file)
        writer.write(model)
        pov_file.write(POV_TRAILER + "\n")


def run_pov(pov_path, image_path, image_width, image_height):
    cmd = [
        "povray",
        f"-i{pov_path}",
        f"+W{image_width}",
        f"+H{image_height}",
        "+FN",
        f"-o{image_path}",
    ]
    result = subprocess.run(cmd, capture_output=True)
    result.check_returncode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outfile", help="Output file for generated image.", required=True
    )
    parser.add_argument(
        "--width",
        help="Width in pixels of output image.",
        default=1024,
        type=int,
    )
    parser.add_argument(
        "--height",
        help="Height in pixels of output image.",
        default=768,
        type=int,
    )
    parser.add_argument(
        "--num_parts",
        help="Number of parts in the pile.",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--num_colors",
        help="Number of colors in the pile.",
        default=4,
        type=int,
    )
    args = parser.parse_args()

    colors = gen_colors(args.num_colors)
    parts = gen_parts(PARTS, colors, args.num_parts)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ldr") as ldr:
        for part in parts:
            ldr.write(str(part) + "\n")
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pov") as pov:
        pov.close()
        gen_pov(ldr.name, pov.name)
        run_pov(pov.name, args.outfile, args.width, args.height)
        print(f"Wrote output file {args.outfile}")


if __name__ == "__main__":
    main()
