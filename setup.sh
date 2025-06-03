#!/usr/bin/env bash
# Setup script run by Codex before network access is disabled.
# Installs the Python dependencies listed in requirements.txt

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
