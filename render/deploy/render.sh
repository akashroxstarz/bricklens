#!/bin/bash

# This script is run inside of the K8s Job container.

set -eux

export WANDB_API_KEY=$(cat /wandb-api-key.txt)

echo "BRICKLENS_TIMESTAMP is: ${BRICKLENS_TIMESTAMP}"
echo "BRICKLENS_JOB_ID is: ${BRICKLENS_JOB_ID}"

echo "Starting render job."

python3 -m bricklens.render.generate_detection_dataset \
  --bricklens_group ${BRICKLENS_TIMESTAMP} \
  --bricklens_job ${BRICKLENS_JOB_ID} \
  --ldraw_library_path /ldraw \
  --outdir /dataset \
  --overwrite \
  --num_images 100 \
  --num_parts 10 \
  --num_colors 4 \
  --background_parts 0 \
  --background_colors 0 \
  --detections_min 10 \
  --detections_max 50 \
  --pile_min 0 \
  --pile_max 0

echo "Copying data to GCS..."

gsutil -m cp -r \
  /dataset \
  gs://bricklens-datasets/renders/${BRICKLENS_TIMESTAMP}/${BRICKLENS_JOB_ID}/

echo "Done."
exit 0
