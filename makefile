#compile flags
# Needed if you are using gcc5, because tf <= 0.12 use gcc4 with old ABI
USE_OLD_ABI ?= true
CC          ?= g++
MKDIR_P     ?= mkdir -p
MODULE_FOLDER ?= DENN
TOP         ?= $(shell pwd)
PYTHON_PATH ?= $(HOME)/.virtualenvs/TensorFlow/
USE_DEBUG   ?= false
OFFICIAL_BINARY ?= true
#include list
TF_INCLUDE = $(PYTHON_PATH)/lib/python3.5/site-packages/tensorflow/include
#Output name
OUT_FILE_NAME_DENNOP = DENNOp
OUT_FILE_NAME_DENNOP_TRAINING = DENNOp_training
OUT_FILE_NAME_DENNOP_ADA = DENNOp_ada
OUT_FILE_NAME_DENNOP_ADA_TRAINING = DENNOp_ada_training
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
	C_FLAGS      += -pthread -D_FORCE_INLINES -fopenmp -DENABLE_PARALLEL_NEW_GEN
	LIKNER_FLAGS += -lpthread -lm -lutil -ldl
	LIKNER_FLAGS += -Wl,--whole-archive 
	LIKNER_FLAGS += -L$(TOP)/tf_static/linux/ -lprotobuf.pic
	LIKNER_FLAGS += -L$(TOP)/tf_static/linux/ -lprotobuf_lite.pic
	LIKNER_FLAGS += -L$(TOP)/tf_static/linux/ -lc_api.pic
	ifeq ($(OFFICIAL_BINARY),true)
		LIKNER_FLAGS += -L$(TOP)/tf_static/linux/ -lloader.pic # Needed only with official binary package
	endif
	# LIKNER_FLAGS += -L$(TOP)/tf_static/ -lpng_custom
	LIKNER_FLAGS += -Wl,--no-whole-archive
endif

#flags macOS
ifeq ($(shell uname -s),Darwin)
	#flags add protobuf
	# LIKNER_FLAGS += -lprotobuf 
	#disable dynamic lookup
	LIKNER_FLAGS += -L$(TOP)/tf_static/macOS/ -lprotobuf.pic
	LIKNER_FLAGS += -L$(TOP)/tf_static/macOS/ -lprotobuf_lite.pic
	C_FLAGS += -undefined dynamic_lookup
endif

#old abi
ifeq ($(USE_OLD_ABI),true)
	C_FLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
endif

#flags C++
ifeq ($(USE_DEBUG),true)
	CPP_FLAGS = -g -D_DEBUG
else
	CPP_FLAGS = -Ofast
endif

#cpp files
SOURCE_FILES = $(S_DIR)/DENNOp.cpp
SOURCE_OBJS = $(addprefix $(O_DIR)/,$(notdir $(SOURCE_FILES:.cpp=.o)))

SOURCE_TRAINING_FILES = $(S_DIR)/DENNOpTraining.cpp
SOURCE_TRAINING_OBJS = $(addprefix $(O_DIR)/,$(notdir $(SOURCE_TRAINING_FILES:.cpp=.o)))

SOURCE_ADA_FILES = $(S_DIR)/DENNOpADA.cpp
SOURCE_ADA_OBJS = $(addprefix $(O_DIR)/,$(notdir $(SOURCE_ADA_FILES:.cpp=.o)))

SOURCE_ADA_TRAINING_FILES = $(S_DIR)/DENNOpAdaTraining.cpp
SOURCE_ADA_TRAINING_OBJS = $(addprefix $(O_DIR)/,$(notdir $(SOURCE_ADA_TRAINING_FILES:.cpp=.o)))
#########################################################################

all: make_denn_op make_denn_traning_op make_denn_ada_op make_denn_ada_traning_op

#create plugin
make_denn_op: directories $(SOURCE_OBJS)
	$(CC) $(C_FLAGS) $(CPP_FLAGS) -shared -o $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP).so $(SOURCE_OBJS) $(LIKNER_FLAGS)

make_denn_traning_op: directories $(SOURCE_TRAINING_OBJS)
	$(CC) $(C_FLAGS) $(CPP_FLAGS) -shared -o $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP_TRAINING).so $(SOURCE_TRAINING_OBJS) $(LIKNER_FLAGS)

make_denn_ada_op: directories $(SOURCE_ADA_OBJS)
	$(CC) $(C_FLAGS) $(CPP_FLAGS) -shared -o $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP_ADA).so $(SOURCE_ADA_OBJS) $(LIKNER_FLAGS)

make_denn_ada_traning_op: directories $(SOURCE_ADA_TRAINING_OBJS)
	$(CC) $(C_FLAGS) $(CPP_FLAGS) -shared -o $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP_ADA_TRAINING).so $(SOURCE_ADA_TRAINING_OBJS) $(LIKNER_FLAGS)
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
	rm -f -R $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP).so
	rm -f -R $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP_TRAINING).so
	rm -f -R $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP_ADA).so
	rm -f -R $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP_ADA_TRAINING).so
