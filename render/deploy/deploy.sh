#!/bin/bash

# This script deploys the renderer Helm chart to the current cluster.

set -eux

# Be sure we are in the right context.
gcloud config configurations activate bricklens
kubectl config use-context gke_bricklens_us-west1_bricklens-cluster

# Create namespace.
export KUBE_NAMESPACE="bricklens-render"

kubectl delete namespace "$KUBE_NAMESPACE"

if ! kubectl get namespace "$KUBE_NAMESPACE"; then
    echo "Creating namespace: $KUBE_NAMESPACE"
    kubectl create namespace "$KUBE_NAMESPACE"
fi

# Create GCR pull secret.
if ! kubectl get secret gcr-deploy-token --namespace="$KUBE_NAMESPACE"; then
    kubectl create secret docker-registry gcr-deploy-token \
        --namespace="$KUBE_NAMESPACE" \
        --docker-server=gcr.io \
        --docker-username=_json_key \
        --docker-password="$(cat secrets/bricklens-c7eddf45242c.json)" \
        --docker-email=any@valid.email
fi

helm dependency update helm/render
helm upgrade --install bricklens-render --namespace "$KUBE_NAMESPACE" ./helm/render \
    --timeout 15m0s --wait --atomic --debug --alsologtostderr

echo "Deployed namespace ${KUBE_NAMESPACE}."
