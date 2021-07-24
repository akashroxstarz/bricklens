#!/bin/bash

gcloud dataproc jobs submit pyspark wordcount.py \
    --cluster=bricklens-spark \
    --region=us-west1 \
    -- gs://bricklens-renders/input/ gs://bricklens-renders/output/
