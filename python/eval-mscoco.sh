#!/bin/sh

poetry run bricklens/train/detection/eval.py \
  --num_images 100 \
  --model_config_path bricklens/train/detection/config/yolov3-mscoco-pjreddie.yml \
  --weights_path bricklens/train/detection/config/yolov3.weights \
  --mscoco_path  ~/datasets/mscoco
