#!/usr/bin/env bash

# Installe les dépendances en local (avant que le réseau ne soit coupé)
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir pytest
# éventuellement : python -m pip install --no-cache-dir -e .
