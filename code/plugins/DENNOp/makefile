#compile flags
USE_OLD_ABI ?= true
CC          ?= g++
MKDIR_P     ?= mkdir -p
MODULE_FOLDER ?= DENN
TOP         ?= $(shell pwd)
PYTHON_PATH ?= $(HOME)/.virtualenvs/TensorFlow_from_source/
USE_DEBUG   ?= false
#include list
TF_INCLUDE = $(PYTHON_PATH)/lib/python3.5/site-packages/tensorflow/include
#project dirs
S_DIR  = $(TOP)/source
S_INC  = $(TOP)/include
O_DIR  = $(TOP)/$(MODULE_FOLDER)/obj
#flags C
C_FLAGS = -Wall -std=c++11 -I $(TF_INCLUDE) -I $(S_INC) -fPIC
#flags protobuf
C_FLAGS += -lprotobuf
#flags C++
ifeq ($(USE_DEBUG),true)
    CPP_FLAGS = -g -D_DEBUG
else
    CPP_FLAGS = -O3
endif
#cpp files
SOURCE_FILES := $(S_DIR)/DENNOp.cpp
SOURCE_OBJS := $(addprefix $(O_DIR)/,$(notdir $(SOURCE_FILES:.cpp=.o)))
#cases
ifeq ($(USE_OLD_ABI),true)
    C_FLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
endif

ifeq "$(shell uname -s)" "Darwin"
    C_FLAGS += -undefined dynamic_lookup
endif

all: make_trial_vector

#create plugin
make_trial_vector: directories $(SOURCE_OBJS)
	$(CC) $(C_FLAGS) $(CPP_FLAGS) -shared -o $(TOP)/$(MODULE_FOLDER)/DENNOp.so $(SOURCE_OBJS)

#required directories
directories: ${O_DIR}

#dir
${O_DIR}:
	${MKDIR_P} ${O_DIR}

#make objects dir
$(O_DIR)/%.o: $(S_DIR)/%.cpp
	$(CC) $(C_FLAGS) $(CPP_FLAGS) -c $< -o $@

#clear dir
clean:
	rm -f -R $(TOP)/$(MODULE_FOLDER)/obj
	rm -f -R $(TOP)/$(MODULE_FOLDER)/*.so
