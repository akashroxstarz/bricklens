#!/bin/bash

# Script run by each K8s Job.

set -eux

echo "BRICKLENS_TIMESTAMP is: ${BRICKLENS_TIMESTAMP}"
echo "BRICKLENS_JOB_ID is: ${BRICKLENS_JOB_ID}"

echo "Starting render job."

python3 -m bricklens.render.generate_detection_dataset \
  --ldraw_library_path /ldraw \
  --outdir /dataset \
  --overwrite \
  --num_images 2

echo "Copying data to GCS..."

gsutil cp -r \
  /dataset \
  gs://bricklens-datasets/renders/${BRICKLENS_TIMESTAMP}/${BRICKLENS_JOB_ID}/

echo "Done."
