#!/bin/sh

set -x

kubectl delete namespace bricklens-render
gcloud container clusters delete bricklens-cluster
