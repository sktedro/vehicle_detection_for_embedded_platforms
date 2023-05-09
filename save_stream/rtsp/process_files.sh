#!/bin/bash

mkdir -p to_process
mkdir -p processed

# Check for mp4 files to process in a cycle until stop is requested
while [ -z $(find ./ -name "stop*") ]; do

    # When a mp4 file to process is found, process it...
    for f in $(find ./to_process/ -name "*.mp4"); do

        # Get the name from the path
        name=$(echo "$f" | rev | cut -d "/" -f 1 | rev | cut -d '.' -f 1)

        echo "Processing: $name"

        mv ./to_process/"$name".mp4 ./processed/"$name"_to_process.mp4
        ffmpeg -i ./processed/"$name"_to_process.mp4 -vcodec libx264 ./processed/"$name".mp4 -loglevel quiet
        echo "Compressed"

        rm ./processed/"$name"_to_process.mp4
        echo "Removed uncompressed"

        rclone copy ./processed/"$name".mp4 remote:path/
        echo "Uploaded"

        rm "./processed/$name.mp4"
        echo "Removed compressed"

        echo "Processed: $name"
        echo
        break # Restart the cycle - search for files again

    done
    sleep 1
done
