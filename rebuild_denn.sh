#!/bin/bash
. $(dirname "$0")/rebuild_util.sh
rm DENN/DENNOp.so
rm DENN/obj/DENNOp.o
make make_denn_op USE_DEBUG=$(debug $@)