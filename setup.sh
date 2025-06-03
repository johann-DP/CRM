#!/usr/bin/env bash

python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir pytest
python -m pip install --no-cache-dir -e .

