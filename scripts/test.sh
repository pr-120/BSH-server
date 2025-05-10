#!/bin/bash

dots=""
while ps aux | grep -P "SCREEN -dmS tick_\d+ bash ./gather_data.sh" >/dev/null 2>&1; do
    dots="$dots."
    if [ "${#dots}" -gt 3 ]; then
        dots="."
    fi
    echo -ne "\rdata still being collected$dots"
    sleep 5
done
echo -e "\rData collection completed.       "

