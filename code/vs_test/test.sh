#!/bin/bash

make clean &&
make TARGET=de_new_gen &&
python3 test_new_op.py