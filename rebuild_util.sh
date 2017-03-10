function debug_flag_test
{
    if [[ "$1" == "debug" ]]; then
        echo "USE_DEBUG=true" 
        else 
        echo "USE_DEBUG=false"
    fi   
}
export -f debug_flag_test