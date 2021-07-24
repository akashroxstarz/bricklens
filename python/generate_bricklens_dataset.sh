#!/bin/sh

poetry run bricklens/render/blender/generate_detection_dataset.py \
  --outdir $HOME/datasets/bricklens_1000 \
  --num_images 1000 \
  --width 512 \
  --height 512 \
  --num_parts 20 \
  --num_colors 20 \
  --background_parts 0 \
  --detections_size 100 \
  --pile_size 0 \
  --frac_train_images 0.7 \
  --overwrite
