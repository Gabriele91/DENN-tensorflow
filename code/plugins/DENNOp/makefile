#compile flags
USE_OLD_ABI ?= true
CC          ?= g++
MKDIR_P     ?= mkdir -p
MODULE_FOLDER ?= DENN
TOP         ?= $(shell pwd)
PYTHON_PATH ?= $(HOME)/.virtualenvs/TensorFlow/
USE_DEBUG   ?= true
#include list
TF_INCLUDE = $(PYTHON_PATH)/lib/python3.5/site-packages/tensorflow/include
#project dirs
S_DIR  = $(TOP)/source
S_INC  = $(TOP)/include
O_DIR  = $(TOP)/$(MODULE_FOLDER)/obj
#flags C
C_FLAGS = -Wall -std=c++11 -I $(TF_INCLUDE) -I $(S_INC) -fPIC
#flags liker
LIKNER_FLAGS =

#flags LINUX
ifeq ($(shell uname -s),Linux)
	#linux flags
	C_FLAGS      += -pthread -D_FORCE_INLINES  
	LIKNER_FLAGS += -lpthread -lm -lutil -ldl
	LIKNER_FLAGS += -Wl,--whole-archive 
	LIKNER_FLAGS += -L$(TOP)/tw_static/ -lprotobuf_custom
	LIKNER_FLAGS += -L$(TOP)/tw_static/ -lprotobuf_lite_custom
	LIKNER_FLAGS += -L$(TOP)/tw_static/ -lpng_custom
	LIKNER_FLAGS += -L$(TOP)/tw_static/ -lc_api
	LIKNER_FLAGS += -L$(TOP)/tw_static/ -lscope
	LIKNER_FLAGS += -Wl,--no-whole-archive 
	#must to be false
	USE_OLD_ABI=false
endif

#flags macOS
ifeq ($(shell uname -s),Darwin)
	#flags add protobuf
	LIKNER_FLAGS += -lprotobuf 
	#disable dynamic lookup
	C_FLAGS += -undefined dynamic_lookup
endif

#cases
ifeq ($(USE_OLD_ABI),true)
	C_FLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
endif

#flags C++
ifeq ($(USE_DEBUG),true)
	CPP_FLAGS = -g -D_DEBUG
else
	CPP_FLAGS = -O3
endif

#cpp files
SOURCE_FILES = $(S_DIR)/DENNOp.cpp
SOURCE_OBJS = $(addprefix $(O_DIR)/,$(notdir $(SOURCE_FILES:.cpp=.o)))

#########################################################################

all: make_trial_vector

#create plugin
make_trial_vector: directories $(SOURCE_OBJS)
	$(CC) $(C_FLAGS) $(CPP_FLAGS) -shared -o $(TOP)/$(MODULE_FOLDER)/DENNOp.so $(SOURCE_OBJS) $(LIKNER_FLAGS) 

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
