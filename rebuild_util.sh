function debug
{
    if [[ $1 -eq "debug" ]]; then
        echo "true" 
        else 
        echo "false"
    fi   
}
export -f debug