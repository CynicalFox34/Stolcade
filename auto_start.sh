#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/lincoln/Documents/stolcade/engine
cd /Users/lincoln/Documents/stolcade
nohup /usr/bin/python3 -u server.py > server.log 2>&1 &
cd /Users/lincoln/Documents/stolcade/engine
nohup /usr/bin/python3 -u train.py > training.log 2>&1 &
echo "Started."
