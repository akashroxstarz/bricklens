#!/bin/sh

poetry run bricklens/train/detection/eval.py \
  --weights_path checkpoints_simple_1000/checkpoint_99.weights \
  --dataset_config_path bricklens/train/detection/config/dataset_detection_simple_1000.yml \
  --slice train
