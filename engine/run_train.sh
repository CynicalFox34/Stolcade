#!/bin/bash
# run_train.sh
# Ensures we are in the right directory and unbuffered output is captured.
cd "$(dirname "$0")"
python3 -u train.py >> training.log 2>&1
