#!/bin/sh

poetry run bricklens/train/detection/train.py \
  --dataset_config_path bricklens/train/detection/config/dataset_detection_simple_1000.yml 
