#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
BRANCH=$1
REMOTE_HOST="beta1l"
REMOTE_DIR="/home/will/projects/vibecoding/Zonos_Vibes" # Use explicit path provided by user
# Use 'uv' directly, assuming PATH is set by sourcing profile
UV_CMD="uv"
# Command used to start the server
SERVER_CMD="$UV_CMD run server.py"
# Pattern used to find the server process (escape special characters for pgrep/pkill)
# Made more specific to avoid matching unrelated processes (matches status script)
SERVER_PATTERN="\.venv/bin/python3 server\.py"
# Log file on the remote server
REMOTE_LOG="server.log"
# --- End Configuration ---

# Check if branch name is provided
if [ -z "$BRANCH" ]; then
  echo "Error: Branch name must be provided as the first argument." >&2
  exit 1
fi

echo "--- Deploying branch '$BRANCH' to $REMOTE_HOST ---"

# Use SSH with a heredoc for clarity and better command handling
# Pass the patterns needed for pkill as arguments ($1, $2) to the remote script
# Use single quotes locally to prevent shell expansion of backslashes
ssh "$REMOTE_HOST" bash -s -- "$SERVER_PATTERN" 'uv run server\.py' << EOF
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
  # Ensure \$HOME is expanded correctly on the remote side
  cd "$REMOTE_DIR" || { echo "[Remote] Error: Failed to change directory to $REMOTE_DIR"; exit 1; }

  # Retrieve patterns from arguments
  ARG_SPECIFIC_PATTERN="\$1"
  ARG_UV_PATTERN="\$2"

  # --- Pre-deployment Check & Stop ---
  echo "[Remote] Checking for running server processes before deployment..."

  echo "[Remote] Attempting to stop processes matching '\$ARG_SPECIFIC_PATTERN' or '\$ARG_UV_PATTERN'..."
  # Use pkill -f with the patterns passed as arguments. Send SIGTERM first, then SIGKILL.
  # The || true prevents the script from exiting if pkill finds no processes.
  pkill -f "\$ARG_SPECIFIC_PATTERN" || true
  pkill -f "\$ARG_UV_PATTERN" || true
  sleep 1 # Give processes a moment to terminate

  # Check if any are still running and send SIGKILL
  if pgrep -f "\$ARG_SPECIFIC_PATTERN" > /dev/null || pgrep -f "\$ARG_UV_PATTERN" > /dev/null; then
      echo "[Remote] Some server processes still running, sending SIGKILL..."
      pkill -9 -f "\$ARG_SPECIFIC_PATTERN" || true
      pkill -9 -f "\$ARG_UV_PATTERN" || true
      sleep 1
  fi

  # Final verification
  if pgrep -f "\$ARG_SPECIFIC_PATTERN" > /dev/null || pgrep -f "\$ARG_UV_PATTERN" > /dev/null; then
      echo "[Remote] Warning: Server processes might still be running after kill attempts."
  else
      echo "[Remote] Server processes stopped successfully."
  fi
  # --- End Pre-deployment Check & Stop ---

  echo "[Remote] Checking out branch: $BRANCH"
  git checkout "$BRANCH"

  echo "[Remote] Discarding potential local changes to server.log..."
  git checkout -- server.log || echo "[Remote] Warning: Failed to discard changes in server.log (maybe it wasn't changed?)"

  echo "[Remote] Pulling latest changes for branch $BRANCH from origin..."
  git pull origin "$BRANCH"

  # Ensure the status script is executable *after* pulling changes
  echo "[Remote] Ensuring status script is executable..."
  chmod +x ./get_status_of_beta1.sh || echo "[Remote] Warning: chmod failed for status script"

  echo "[Remote] Syncing dependencies with uv..."
  $UV_CMD sync || { echo "[Remote] Error: uv sync failed."; exit 1; }

  echo "[Remote] Attempting background server start with nohup: $SERVER_CMD"
  # Start in background, redirect stdout/stderr to log file
  nohup $SERVER_CMD > "$REMOTE_LOG" 2>&1 &
  # Capture the PID of the newly started process
  NEW_PID=\$!

  # Brief pause to allow server to start/fail
  sleep 2

  # Verify the server process started
  if kill -0 \$NEW_PID 2>/dev/null; then
      echo "[Remote] Server started successfully (PID: \$NEW_PID). Output logged to $REMOTE_LOG"
  else
      echo "[Remote] Error: Server failed to start. Check $REMOTE_LOG on $REMOTE_HOST for details."
      exit 1 # Exit with error if server didn't start
  fi

  echo "[Remote] Deployment steps completed."
EOF

# Check the exit status of the SSH command itself
SSH_EXIT_STATUS=$?
if [ $SSH_EXIT_STATUS -ne 0 ]; then
  echo "--- Error: SSH command failed with exit status $SSH_EXIT_STATUS ---" >&2
  exit $SSH_EXIT_STATUS
fi

echo "--- Deployment script finished successfully ---"
exit 0
