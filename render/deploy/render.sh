#!/bin/bash

# This script is run inside of the K8s Job container.

set -eux

echo "BRICKLENS_TIMESTAMP is: ${BRICKLENS_TIMESTAMP}"
echo "BRICKLENS_JOB_ID is: ${BRICKLENS_JOB_ID}"

echo "Starting render job."

python3 -m bricklens.render.generate_detection_dataset \
  --ldraw_library_path /ldraw \
  --outdir /dataset \
  --overwrite \
  --num_images 100 \
  --num_parts 10 \
  --num_colors 4 \
  --background_parts 4 \
  --background_colors 4 \
  --detections_min 1 \
  --detections_max 20 \
  --pile_min 5 \
  --pile_max 200

echo "Copying data to GCS..."

gsutil cp -r \
  /dataset \
  gs://bricklens-datasets/renders/${BRICKLENS_TIMESTAMP}/${BRICKLENS_JOB_ID}/

echo "Done."
