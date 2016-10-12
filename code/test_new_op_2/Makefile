TARGET ?= example.cc
GCCVERSION ?= 5
TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CC = g++
CFLAGS = -Wall -std=c++11 -I $(TF_INC) -fPIC 

ifeq "$(GCCVERSION)" "5"
    CFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
endif

ifeq "$(shell uname -s)" "Darwin"
    CFLAGS += -undefined dynamic_lookup
endif

all: operation

operation: $(TARGET)
	$(CC) $(CFLAGS) -shared $(TARGET) -o $(TARGET:.cc=.so)
clean:
	rm -f $(TARGET:.cc=.so)