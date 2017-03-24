# Compiler flags
# Needed if you are using gcc5, because tf <= 0.12 use gcc4 with old ABI
USE_OLD_ABI ?= true
CC          ?= g++
MKDIR_P     ?= mkdir -p
MODULE_FOLDER ?= DENN
TOP         ?= $(shell pwd)
USE_DEBUG   =
REBUILD_OP  =
#include list
TF_INCLUDE = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())' 2> /dev/null)
#Output name
OUT_FILE_NAME_DENNOP = DENNOp
OUT_FILE_NAME_DENNOP_TRAINING = DENNOpTraining
OUT_FILE_NAME_DENNOP_ADA = DENNOpADA
OUT_FILE_NAME_DENNOP_ADA_TRAINING = DENNOpAdaTraining
#project dirs
S_DIR  = $(TOP)/source
S_INC  = $(TOP)/include
O_DIR  = $(TOP)/$(MODULE_FOLDER)/obj
# C FLAGS
C_FLAGS = -Wall -std=c++11 -I $(TF_INCLUDE) -I $(S_INC) -fPIC
# Linker FLAGS
LIKNER_FLAGS =
# CPP flags 
CPP_FLAGS =
# Pass from other make file 
MAKE_PARAMS = 
# Color Types
COLOR_BLACK = 0
COLOR_RED = 1
COLOR_GREEN = 2
COLOR_YELLOW = 3
COLOR_BLUE = 4
COLOR_MAGENTA = 5
COLOR_CYAN = 6
COLOR_WHITE = 7

##
# Support function for colored output
# Args:
#     - $(1) = Color Type
#     - $(2) = String to print
define colorecho
      @tput setaf $(1)
      @echo $(2)
      @tput sgr0
endef

# LINUX FLAGS
ifeq ($(shell uname -s),Linux)
	#linux flags
	C_FLAGS      += -pthread -D_FORCE_INLINES -fopenmp -DENABLE_PARALLEL_NEW_GEN
	LIKNER_FLAGS += -lpthread -lm -lutil -ldl
	LIKNER_FLAGS += -Wl,--whole-archive 
	LIKNER_FLAGS += -L$(TOP)/tf_static/linux/ -lprotobuf.pic
	LIKNER_FLAGS += -L$(TOP)/tf_static/linux/ -lprotobuf_lite.pic
	LIKNER_FLAGS += -L$(TOP)/tf_static/linux/ -lc_api.pic
	# LIKNER_FLAGS += -L$(TOP)/tf_static/ -lpng_custom
	LIKNER_FLAGS += -Wl,--no-whole-archive
endif

# MacOS FLAGS
ifeq ($(shell uname -s),Darwin)
	#flags add protobuf
	# LIKNER_FLAGS += -lprotobuf 
	#disable dynamic lookup
	LIKNER_FLAGS += -L$(TOP)/tf_static/macOS/ -lprotobuf.pic
	LIKNER_FLAGS += -L$(TOP)/tf_static/macOS/ -lprotobuf_lite.pic
	C_FLAGS += -undefined dynamic_lookup -Wno-deprecated-declarations
endif

# Old ABI
ifeq ($(USE_OLD_ABI),true)
	C_FLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
endif

# C++ files
SOURCE_FILES = $(S_DIR)/DENNOp.cpp
SOURCE_OBJS = $(addprefix $(O_DIR)/,$(notdir $(SOURCE_FILES:.cpp=.o)))

SOURCE_TRAINING_FILES = $(S_DIR)/DENNOpTraining.cpp
SOURCE_TRAINING_OBJS = $(addprefix $(O_DIR)/,$(notdir $(SOURCE_TRAINING_FILES:.cpp=.o)))

SOURCE_ADA_FILES = $(S_DIR)/DENNOpADA.cpp
SOURCE_ADA_OBJS = $(addprefix $(O_DIR)/,$(notdir $(SOURCE_ADA_FILES:.cpp=.o)))

SOURCE_ADA_TRAINING_FILES = $(S_DIR)/DENNOpAdaTraining.cpp
SOURCE_ADA_TRAINING_OBJS = $(addprefix $(O_DIR)/,$(notdir $(SOURCE_ADA_TRAINING_FILES:.cpp=.o)))
#########################################################################

# Support function to delete a single Op
define delete_op
	$(call colorecho,$(COLOR_GREEN),"[ Remove $(1).o and $(1).so files ]")
    @rm -f "$(TOP)/$(MODULE_FOLDER)/$(1).so"
	@rm -f "$(TOP)/$(MODULE_FOLDER)/obj/$(1).o"
endef

all: denn denn_traning denn_ada denn_ada_traning

.PHONY: rebuild debug all denn denn_traning denn_ada denn_ada_traning directories ${O_DIR} clean

# Set rebuild option
rebuild:
	$(call colorecho,$(COLOR_YELLOW),"[ Set Rebuild ]")
	$(eval REBUILD_OP = true)

# Set debug flags
debug:
	$(call colorecho,$(COLOR_MAGENTA),"[ Set DEBUG ]")
	$(eval USE_DEBUG = true)
	
# Create plugin
denn: directories 
# if REBUILD_OP is not an empy string the condition is true and
# function will be called
	$(if $(REBUILD_OP),$(call delete_op,$(OUT_FILE_NAME_DENNOP)))
	$(if $(USE_DEBUG),$(eval MAKE_PARAMS=-g -D_DEBUG),$(eval MAKE_PARAMS=-Ofast))
	$(MAKE) $(SOURCE_OBJS) CC=$(CC) C_FLAGS="$(C_FLAGS)" CPP_FLAGS="$(CPP_FLAGS)" MAKE_PARAMS="$(MAKE_PARAMS)"
	$(call colorecho,$(COLOR_GREEN),"[ Make $(OUT_FILE_NAME_DENNOP).so ]")
	$(CC) $(C_FLAGS) $(CPP_FLAGS) $(MAKE_PARAMS) -shared -o $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP).so $(SOURCE_OBJS) $(LIKNER_FLAGS)

denn_traning: directories
	$(if $(REBUILD_OP),$(call delete_op,$(OUT_FILE_NAME_DENNOP_TRAINING)))
	$(if $(USE_DEBUG),$(eval MAKE_PARAMS=-g -D_DEBUG),$(eval MAKE_PARAMS=-Ofast))
	$(MAKE) $(SOURCE_TRAINING_OBJS) CC=$(CC) C_FLAGS="$(C_FLAGS)" CPP_FLAGS="$(CPP_FLAGS)" MAKE_PARAMS="$(MAKE_PARAMS)"
	$(call colorecho,$(COLOR_GREEN),"[ Make $(OUT_FILE_NAME_DENNOP_TRAINING).so ]")
	$(CC) $(C_FLAGS) $(CPP_FLAGS) $(MAKE_PARAMS) -shared -o $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP_TRAINING).so $(SOURCE_TRAINING_OBJS) $(LIKNER_FLAGS)

denn_ada: directories
	$(if $(REBUILD_OP),$(call delete_op,$(OUT_FILE_NAME_DENNOP_ADA)))
	$(if $(USE_DEBUG),$(eval MAKE_PARAMS=-g -D_DEBUG),$(eval MAKE_PARAMS=-Ofast))
	$(MAKE) $(SOURCE_ADA_OBJS) CC=$(CC) C_FLAGS="$(C_FLAGS)" CPP_FLAGS="$(CPP_FLAGS)" MAKE_PARAMS="$(MAKE_PARAMS)"
	$(call colorecho,$(COLOR_GREEN),"[ Make $(OUT_FILE_NAME_DENNOP_ADA).so ]")
	$(CC) $(C_FLAGS) $(CPP_FLAGS) $(MAKE_PARAMS) -shared -o $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP_ADA).so $(SOURCE_ADA_OBJS) $(LIKNER_FLAGS)

denn_ada_traning: directories
	$(if $(REBUILD_OP),$(call delete_op,$(OUT_FILE_NAME_DENNOP_ADA_TRAINING)))
	$(if $(USE_DEBUG),$(eval MAKE_PARAMS=-g -D_DEBUG),$(eval MAKE_PARAMS=-Ofast))
	$(MAKE) $(SOURCE_ADA_TRAINING_OBJS) CC=$(CC) C_FLAGS="$(C_FLAGS)" CPP_FLAGS="$(CPP_FLAGS)" MAKE_PARAMS="$(MAKE_PARAMS)"
	$(call colorecho,$(COLOR_GREEN),"[ Make $(OUT_FILE_NAME_DENNOP_ADA_TRAINING).so ]")
	$(CC) $(C_FLAGS) $(CPP_FLAGS) $(MAKE_PARAMS) -shared -o $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP_ADA_TRAINING).so $(SOURCE_ADA_TRAINING_OBJS) $(LIKNER_FLAGS)

# Required directories
directories: ${O_DIR}

# makedir
${O_DIR}:
	$(call colorecho,$(COLOR_GREEN),"[ Create $(O_DIR) directory ]")
	@${MKDIR_P} ${O_DIR}

# Make objects files
$(O_DIR)/%.o: $(S_DIR)/%.cpp
	$(call colorecho,$(COLOR_GREEN),"[ Make object $(@) ]")
	$(CC) $(C_FLAGS) $(CPP_FLAGS) $(MAKE_PARAMS) -c $< -o $@

# Clean
clean:
	$(call colorecho,$(COLOR_MAGENTA),"[ Delete obj files ]")
	@rm -f -R $(TOP)/$(MODULE_FOLDER)/obj
	$(call colorecho,$(COLOR_MAGENTA),"[ Delete DENNOp ]")
	@rm -f -R $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP).so
	$(call colorecho,$(COLOR_MAGENTA),"[ Delete DENNOp Training ]")
	@rm -f -R $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP_TRAINING).so
	$(call colorecho,$(COLOR_MAGENTA),"[ Delete DENNOp ADA ]")
	@rm -f -R $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP_ADA).so
	$(call colorecho,$(COLOR_MAGENTA),"[ Delete DENNOp ADA Training ]")
	@rm -f -R $(TOP)/$(MODULE_FOLDER)/$(OUT_FILE_NAME_DENNOP_ADA_TRAINING).so
