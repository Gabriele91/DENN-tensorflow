#!/bin/bash
. $(dirname "$0")/rebuild_util.sh
rm DENN/DENNOp_ada.so
rm DENN/obj/DENNOpADA.o
make make_denn_ada_op $(debug_flag_test $@)
