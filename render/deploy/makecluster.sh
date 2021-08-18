#!/bin/sh

# I should probably use Terraform for this.

gcloud config set project bricklens
gcloud config set compute/zone us-west1-a
gcloud config set compute/region us-west1

# This makes a standard cluster.
# gcloud container clusters create bricklens-cluster --num-nodes=1

gcloud beta container \
  --project "bricklens" \
  clusters create-auto "bricklens-cluster" \
  --region "us-west1" \
  --release-channel "regular" \
  --network "projects/bricklens/global/networks/default" \
  --subnetwork "projects/bricklens/regions/us-west1/subnetworks/default" \
  --cluster-ipv4-cidr "/17" \
  --services-ipv4-cidr "/22"

gcloud container clusters get-credentials bricklens-cluster --region us-west1
