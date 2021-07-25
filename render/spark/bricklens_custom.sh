#! /usr/bin/bash

# Script to create Bricklens rendering environment on Dataproc.

apt-get -y update
apt-get install python-dev
apt-get install python-pip
pip install numpy