#!/bin/bash

# --- Configuration (should match deploy script) ---
REMOTE_HOST="beta1l"
REMOTE_DIR="/home/will/projects/vibecoding/Zonos_Vibes"
# Use 'uv' directly, assuming PATH is set by sourcing profile
UV_CMD="uv"
# Command used to start the server
SERVER_CMD="$UV_CMD run server.py"
# Pattern used to find the server process (escape special characters for pgrep/pkill)
# Updated to match the actual process command found via 'ps aux'
SERVER_PATTERN="server\.py"
# --- End Configuration ---

echo "--- Checking status of server on $REMOTE_HOST ---"

# Use SSH to execute commands remotely
ssh "$REMOTE_HOST" bash -s << EOF
  # Attempt to source profile to get PATH settings
  if [ -f ~/.profile ]; then
    echo "[Remote] Sourcing ~/.profile..."
    . ~/.profile
  elif [ -f ~/.bash_profile ]; then
    echo "[Remote] Sourcing ~/.bash_profile..."
    . ~/.bash_profile
  elif [ -f ~/.bashrc ]; then
     echo "[Remote] Sourcing ~/.bashrc..."
     . ~/.bashrc
  else
     echo "[Remote] Warning: No profile script found to source (.profile, .bash_profile, .bashrc)."
  fi

  # Exit immediately if a command exits with a non-zero status on the remote host
  set -e

  echo "[Remote] Changing to directory: $REMOTE_DIR"
  cd "$REMOTE_DIR" || { echo "[Remote] Error: Failed to change directory to $REMOTE_DIR"; exit 1; }

  echo "[Remote] Checking for running server process matching: $SERVER_PATTERN"
  # Find the PID, redirect stderr to /dev/null, ignore exit code if not found
  PID=\$(pgrep -f "$SERVER_PATTERN" 2>/dev/null) || true

  if [ -n "\$PID" ]; then
    echo "[Remote] Server is RUNNING (PID: \$PID)."
    exit 0 # Exit with success code if running
  else
    echo "[Remote] Server is NOT RUNNING."
    exit 1 # Exit with error code if not running
  fi
EOF

# Check the exit status of the SSH command
SSH_EXIT_STATUS=$?
if [ $SSH_EXIT_STATUS -eq 0 ]; then
  echo "--- Server Status: RUNNING ---"
  exit 0
elif [ $SSH_EXIT_STATUS -eq 1 ]; then
  echo "--- Server Status: NOT RUNNING ---"
  exit 0 # Exit successfully even if not running, the script's purpose is to report status
else
  echo "--- Error: SSH command failed or unexpected remote exit status ($SSH_EXIT_STATUS) ---" >&2
  exit $SSH_EXIT_STATUS # Propagate other errors
fi
