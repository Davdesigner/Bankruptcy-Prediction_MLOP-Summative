#!/usr/bin/env bash
apt-get update && apt-get install -y gfortran build-essential libatlas-base-dev
pip install --upgrade pip
pip install -r requirements.txt
