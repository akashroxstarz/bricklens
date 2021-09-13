#!/bin/sh

# This script creates an autopilot GKE cluster.

# Be sure to 'gcloud auth login' first.
gcloud config configurations activate bricklens
gcloud config set project bricklens
gcloud config set compute/zone us-west1-a
gcloud config set compute/region us-west1

# Required to enable private clusters to get Internet access.
gcloud compute routers create bricklens-nat-router \
    --project=bricklens \
    --network projects/bricklens/global/networks/default \
    --region us-west1

gcloud compute routers nats create bricklens-nat-config \
    --project=bricklens \
    --region us-west1 \
    --router bricklens-nat-router \
    --nat-all-subnet-ip-ranges \
    --auto-allocate-nat-external-ips

gcloud beta container \
  --project "bricklens" \
  clusters create-auto "bricklens-cluster" \
  --region "us-west1" \
  --release-channel "regular" \
  --network "projects/bricklens/global/networks/default" \
  --subnetwork "projects/bricklens/regions/us-west1/subnetworks/default" \
  --enable-private-nodes
#  --cluster-ipv4-cidr "/17" \
#  --services-ipv4-cidr "/22"

gcloud container clusters get-credentials bricklens-cluster --region us-west1
kubectl config use-context gke_bricklens_us-west1_bricklens-cluster
