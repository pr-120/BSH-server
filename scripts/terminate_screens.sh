#!/bin/bash

# Check if at least one port is provided
if [ $# -eq 0 ]; then
    printf "Error: No ports provided. Usage: %s port1 [port2 ...]\n" "$0"
    exit 1
fi

printf "\n--> Killing tick sessions\n"

# Flag to track if any sessions were found
any_sessions_found=false

# Loop through all provided ports
for port in "$@"; do

    # Validate port is a number
    if ! [[ "$port" =~ ^[0-9]+$ ]]; then
        printf "Warning: '%s' is not a valid port number, skipping\n" "$port"
        continue
    fi

    printf "\nProcessing port: %s\n" "$port"

    # List SCREEN sessions and filter for those containing "tick_$port"
    screens=$(screen -ls | grep "tick_$port" | awk '{print $1}')

    if [ -z "$screens" ]; then
        printf "No SCREEN sessions with 'tick_%s' found\n" "$port"
        continue
    fi

    # Mark that at least one session was found
    any_sessions_found=true

    # Extract process IDs from screen session names and kill them
    for session in $screens; do

        # Get PID from screen session
        pid=$(screen -ls | grep "$session" | awk '{print $1}' | cut -d. -f1)

        if [ -n "$pid" ]; then
            printf "Killing SCREEN session %s (PID: %s)\n" "$session" "$pid"
            kill "$pid" 2>/dev/null || printf "Failed to kill PID %s\n" "$pid"
        else
            printf "Warning: Could not extract PID for session %s\n" "$session"
        fi
    done
done


# Print completion message
if [ "$any_sessions_found" = true ]; then
    printf "\nDone\n"
else
    printf "\nNo sessions were found for any provided ports\n"
fi

exit 0

