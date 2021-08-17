#!/bin/sh

# I should probably use Terraform for this.

gcloud config set project bricklens
gcloud config set compute/zone us-west1-a
gcloud config set compute/region us-west1

gcloud container clusters create bricklens-cluster --num-nodes=1
gcloud container clusters get-credentials bricklens-cluster

