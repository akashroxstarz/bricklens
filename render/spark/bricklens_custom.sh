#! /usr/bin/bash

# Script to create Bricklens rendering environment on Dataproc.

curl -O https://github.com/TobyLobster/ImportLDraw/releases/download/1.1.11/importldraw1.1.11_for_blender_281.zip

apt-get -y update
apt-get install python-dev
apt-get install python-pip
pip install numpy
