#!/bin/bash

set -eux

# Create namespace.
export KUBE_NAMESPACE="bricklens-render"

if ! kubectl get namespace "$KUBE_NAMESPACE"; then
    info "Creating namespace: $KUBE_NAMESPACE"
    kubectl create namespace "$KUBE_NAMESPACE"
fi


# Create GCR pull secret.
if ! kubectl get secret gcr-deploy-token --namespace="$KUBE_NAMESPACE"; then
    kubectl create secret docker-registry gcr-deploy-token \
        --docker-server=gcr.io \
        --docker-username=_json_key \
        --docker-password="$(cat secrets/bricklens-c7eddf45242c.json)" \
        --docker-email=any@valid.email
fi

helm dependency update deploy/helm/render
helm upgrade --install bricklens-render --namespace "$KUBE_NAMESPACE" ./deploy/helm/render \
    --timeout 15m0s --wait --atomic --debug --alsologtostderr

echo "Deployed namespace ${KUBE_NAMESPACE}."
