#!/bin/bash

# Update pip
pip install --upgrade pip

# Install numpy first (required for other packages)
pip install numpy==1.26.4

# Install scipy with no-cache to avoid build issues
pip install --no-cache-dir scipy==1.11.4

# Install remaining requirements
pip install -r requirements.txt
