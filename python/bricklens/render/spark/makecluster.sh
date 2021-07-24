#!/bin/sh

PROJECT=bricklens
BUCKET_NAME=bricklens-renders
CLUSTER=bricklens-spark
REGION=us-west1

export GOOGLE_APPLICATION_CREDENTIALS=$HOME/keys/bricklens-d1b12863940e.json

gcloud dataproc clusters create ${CLUSTER} \
    --project=${PROJECT} \
    --region=${REGION} \
    --single-node