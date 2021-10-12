# Training

## Dataset format

Dataset format is defined as follows:
```
datasets/bricklens/
   classes.txt
   train.txt
   val.txt
   images/
      image_00000.png
      image_00001.png
   labels/
      image_00000.txt
      image_00001.txt
```

`classes.txt` is of the format:

```
0 4201_Black
1 4201_Blue
```

`train.txt` and `val.txt` are of the format:

```
./images/image_00000.png
./images/image_00001.png
./images/image_00002.png
```

The location of the corresponding label file is inferred based on the image filename.

Each label file is of the format:

```
108 0.245117 0.551758 0.091797 0.095703
39 0.694336 0.392578 0.076172 0.062500
27 0.322266 0.082031 0.042969 0.078125
```
which is:
```
<class-index> <box-center-x> <box-center-y> <box-width> <box-height>
```
where the x, y, width, height values are normalized to the image width and height
(that is, between 0 and 1).

## Dataset generation

### Running locally

```
$ cd bricklens/render
$ just install

$ poetry run bricklens/render/generate_detection_dataset.py \
  --outdir foo \
  --overwrite \
  --num_images 2 \
  --ldraw_library_path $HOME/src/downloads/ldraw
```

### Running on GCS

```
# Edit render.sh with desired flags.
$ vim bricklens/render/deploy/render.sh

# Create Docker image with all dependencies.
$ cd bricklens/render
$ just push-docker

# Create autopilot GKE cluster.
$ cd bricklens/render/deploy
$ ./makecluster.sh

# Deploy Helm chart.
$ ./deploy.sh
```
The resulting dataset will be in
`gs://bricklens-datasets/renders/${BRICKLENS_TIMESTAMP}/${BRICKLENS_JOB_ID}/`.

Run `bricklens/render/deploy/teardown.sh` to tear down the GKE cluster.


## Model training

Based on a fork of: https://github.com/ultralytics/yolov3

```
# Activate Python venv
$ . $HOME/venvs/yolov3/bin/activate

# Run training script
$ cd bricklens/train/yolov3/
$ ./mdw-train.sh
```

Results will be left in `/bricklens/train/yolov3/runs/train/expNNNN`.

## Testing model

```
# Activate Python venv
$ . $HOME/venvs/yolov3/bin/activate

# Run training script
$ cd bricklens/train/yolov3/
$ ./mdw-detect.sh
```

Image detections will be left in `/bricklens/train/yolov3/runs/detect/expNNNN`.

