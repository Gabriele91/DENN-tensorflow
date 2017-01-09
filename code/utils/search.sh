#!/bin/bash

##
# Use example
# 
#     ./search.sh $(pwd) '*.a' tensorflow14LoadSavedModelERKNS_14SessionOptions
#

SOURCE_FOLDER=$1
FILE_FILTER=$2
STRING_TO_FIND=$3

# echo $SOURCE_FOLDER $STRING_TO_FIND

cd $SOURCE_FOLDER;
FILES=($(find -name $FILE_FILTER | sort))


# echo ${FILES[*]}

for cur_file in "${FILES[@]}"
do
    echo $(objdump -t $cur_file | grep $STRING_TO_FIND)
    objdump -t $cur_file | grep -q $STRING_TO_FIND;
    if [ $(echo $?) == "0" ]; then
        echo "+ $cur_file"
    fi
done