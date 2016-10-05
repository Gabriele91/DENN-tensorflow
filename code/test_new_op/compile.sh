#!/bin/bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') &&
g++ -Wall -std=c++11 -shared zero_out.cc -o zero_out.so -I $TF_INC -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
