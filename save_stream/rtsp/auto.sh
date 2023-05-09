#!/bin/sh

# Process files in the background
sh process_files.sh &
pid="$!"

# In case of CTRL+C, kill the process_files.sh and exit
trap ctrl_c INT
function ctrl_c(){
    kill "$pid"
    exit 1
}

# Continously run save_stream.py, unless stop is requested
while [ -z $(find ./ -name "stop*") ]; do 
    python3 save_stream.py
done

# If the process_files.sh is still running, wait for it to finish
if [ ps -p $pid > /dev/null ]; then
    echo "process_files.sh is still running. Waiting. To cancel, hit CTRL+C"
    while [ ps -p $pid > /dev/null ]; do
        sleep 1
    done
fi

# Remove the "stop*" file
rm $(find ./ -name "stop*")
