#!/usr/bin/env bash
# Setup script run by Codex before network access is disabled.
# Installs additional CLI utilities and the Python dependencies listed in
# requirements.txt

# Install some common editors and monitoring tools
apt-get update && apt-get install -y nano vim htop less

# Install Python packages
python -m pip install -r requirements.txt
