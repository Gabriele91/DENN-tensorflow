#!/bin/bash

mkdir -p logs

RED='\033[1;31m'
PURPLE='\033[1;35m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "$RED!!! Executing $PURPLE$GREEN[$PURPLE "$@" $GREEN]$RED with nohup !!!$NC"

TMP_NAME=$(echo "$2" | tr / _)
FILE_NAME=(${TMP_NAME//./ })
OUT_FILE_NAME=${FILE_NAME[0]}_$(date +%s).out

##
# nohup python $@ &> logs/$OUT_FILE_NAME &

##
# Thest with inhibit from Python of SIGHUP
python $@ &> logs/$OUT_FILE_NAME &
