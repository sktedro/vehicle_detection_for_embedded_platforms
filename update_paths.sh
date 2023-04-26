#!/bin/bash

in_dir(){
    for f in "$1"/*; do
        if [ -f "$f" ]; then
            if [ ${f##*/} = "last_checkpoint" ] || [ ${f##*.} = "py" ]; then
                if [ -z "$2" ] || [ -z "$3" ]; then
                    echo "$f"
                else
                    echo "Processing $f"
                    CMD="s~$2~$3~g"
                    # echo $CMD
                    sed -i "$CMD" "$f"
                fi
            fi
        fi
    done
}

echo "Changing all occurances of $1 to $2 in these files:"

in_dir "."
in_dir "configs"
in_dir "deploy"
for d in ./work*; do
    if [ -d "$d" ]; then
        in_dir "$d"
    fi
done

read -p "Are you sure? (y/n)" choice
case "$choice" in
  y)
    in_dir "." "$1" "$2"
    in_dir "configs" "$1" "$2"
    in_dir "deploy" "$1" "$2"
    for d in ./work*; do
        if [ -d "$d" ]; then
            in_dir "$d" "$1" "$2"
        fi
    done
    ;;
  *)
    echo "cancelling"
    ;;
esac
