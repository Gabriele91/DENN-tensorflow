#!/bin/bash

touch daemon.pid ;
PID=$(cat daemon.pid);

if [[ $1 == "status" ]]; then
    kill -0 $PID > /dev/null 2> /dev/null ;
    RETURN=$?
    if [[ $RETURN == 0 ]]; then
        echo "+ Process is running with pid $PID!";
    else
        echo "+ Process with $PID is not running!";
    fi
elif [[ $1 == "start" ]]; then
    python tf_bot.py &> std.out &
    echo $! > daemon.pid ;
    PID=$(cat daemon.pid);
    echo "+ Process started with pid $PID!";
elif [[ $1 == "stop" ]]; then
    kill -15 $PID 2> /dev/null ;
    echo "+ Process with $PID stopped!";
else
    echo "+ Error: Not a valid command!";
fi
