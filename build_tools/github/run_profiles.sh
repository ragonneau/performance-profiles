#!/usr/bin/env bash

set -e
set -x

python main.py
find archives -type f -name "*.csv" -exec rm -rf {} \;
