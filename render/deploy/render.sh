#!/bin/bash

set -eux

echo "Starting render job..."

python3 -m bricklens.render.generate_detection_dataset \
  --ldraw_library_path /ldraw \
  --outdir /dataset \
  --overwrite \
  --num_images 2

echo "Done."
