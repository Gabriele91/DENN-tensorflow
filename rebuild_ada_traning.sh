#!/bin/bash
. $(dirname "$0")/rebuild_util.sh
rm DENN/DENNOp_ada_training.so
rm DENN/obj/DENNOpAdaTraining.o  
make make_denn_ada_traning_op USE_DEBUG=$(debug $@)