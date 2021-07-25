#! /usr/bin/bash

# Script to create Bricklens rendering environment on Dataproc.

set -eux

# Install APT dependencies.
apt-get -y update && \
  apt-get install python3-dev blender

# Install poetry.
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# Install Bricklens library.
git clone https://github.com/mdwelsh/bricklens.git
cd bricklens/render
poetry install && poetry build
pip install dist/bricklens_render-0.1.0-py3-none-any.whl

# TODO(mdw): Figure out how to install this from the commandline.
# curl -O https://github.com/TobyLobster/ImportLDraw/releases/download/1.1.11/importldraw1.1.11_for_blender_281.zip

# Download LDRAW library.
curl -O https://www.ldraw.org/library/updates/complete.zip
unzip complete.zip

python -m bricklens.render.generate_detection_dataset --help
