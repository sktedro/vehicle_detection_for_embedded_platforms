#!/bin/bash

mkdir -p "done"

# Check for mp4 files to process in a cycle until stop is requested
while [ -z $(find ./ -name "stop_p*") ]; do
    # if [ -d "done" ] && [ $(ls -A done) ]; then
    # fi

    # When a mp4 file to process is found, process it...
    for f in $(find ./done/ -name "*.mp4"); do

        path="$(dirname $f)"
        dirname="$(basename $path)"

        echo "Uploading: $f"
        rclone copy "$f" vutbrdrive:traffic_cams/"$dirname"
        echo "Uploaded: $f"
        rm "$f"
        echo "Removed: $f"

        break # Restart the cycle - search for files again

    done
    sleep 1
done
