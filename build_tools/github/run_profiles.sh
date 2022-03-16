#!/usr/bin/env bash

set -e
set -x

source venv/bin/activate
python main.py
find archives -type f -name "*.csv" -exec rm -rf {} \;
