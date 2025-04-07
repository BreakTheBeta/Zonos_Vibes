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
SERVER_PATTERN="$UV_CMD run server\.py"
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
  # Ensure \$HOME is expanded correctly on the remote side
  cd "$REMOTE_DIR" || { echo "[Remote] Error: Failed to change directory to $REMOTE_DIR"; exit 1; }

  echo "[Remote] Checking for running server process matching: $SERVER_PATTERN"
  # Find the PID, redirect stderr to /dev/null, ignore exit code if not found
  PID=\$(pgrep -f "$SERVER_PATTERN" 2>/dev/null) || true

  if [ -n "\$PID" ]; then
    echo "[Remote] Found running server (PID: \$PID). Attempting to stop..."
    # Try to kill the process gracefully first, then forcefully if needed
    kill "\$PID" 2>/dev/null || true
    sleep 1 # Give it a moment to shut down
    # Check if it's still running
    if kill -0 "\$PID" 2>/dev/null; then
        echo "[Remote] Server \$PID still running, sending SIGKILL..."
        kill -9 "\$PID" 2>/dev/null || true
        sleep 1
    fi
    # Verify it's stopped
    if pgrep -f "$SERVER_PATTERN" > /dev/null; then
       echo "[Remote] Warning: Server process might still be running after kill attempts."
    else
       echo "[Remote] Server stopped successfully."
    fi
  else
    echo "[Remote] Server not found running."
  fi

  echo "[Remote] Checking out branch: $BRANCH"
  git checkout "$BRANCH"

  echo "[Remote] Pulling latest changes for branch $BRANCH from origin..."
  git pull origin "$BRANCH"

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
