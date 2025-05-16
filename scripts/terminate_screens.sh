#!/bin/bash

printf "\n--> Killing tick sessions\n"

# List SCREEN sessions and filter for those containing 'tick_'
screens=$(screen -ls | grep "tick_" | awk '{print $1}')

if [ -z "$screens" ]; then
    printf "No SCREEN sessions with 'tick_' found\n"
    exit 0
fi

# Extract process IDs from screen session names and kill them
for session in $screens; do
    # Get PID from screen session
    pid=$(screen -ls | grep "$session" | awk '{print $1}' | cut -d. -f1)
    if [ -n "$pid" ]; then
        printf "Killing SCREEN session %s (PID: %s)\n" "$session" "$pid"
        kill "$pid" 2>/dev/null || printf "Failed to kill PID %s\n" "$pid"
    fi
done

printf "Done\n"
