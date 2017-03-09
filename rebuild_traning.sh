#!/bin/bash
. $(dirname "$0")/rebuild_util.sh
rm DENN/DENNOp_training.so
rm DENN/obj/DENNOpTraining.o
make make_denn_traning_op $(debug_flag_test $@)
