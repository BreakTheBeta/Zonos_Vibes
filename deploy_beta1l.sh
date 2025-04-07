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
# Updated to match the actual process command found via 'ps aux'
SERVER_PATTERN="server\.py"
# Log file on the remote server
REMOTE_LOG="server.log"
# --- End Configuration ---

# Check if branch name is provided
if [ -z "$BRANCH" ]; then
  echo "Error: Branch name must be provided as the first argument." >&2
  exit 1
fi

echo "--- Deploying branch '$BRANCH' to $REMOTE_HOST ---"

# Check current server status first using the dedicated script
echo "--- Checking remote server status before deployment ---"
# Make sure the status script is executable locally (might be needed if cloned fresh)
chmod +x ./get_status_of_beta1.sh || echo "Warning: chmod failed for status script"
./get_status_of_beta1.sh
STATUS_EXIT_CODE=$? # Capture exit code (0=running, 1=not running, >1=error)
echo "--- Status script exited with code: $STATUS_EXIT_CODE ---"

# Handle potential errors from the status script itself
if [ $STATUS_EXIT_CODE -gt 1 ]; then
    echo "Error: Status check script failed with exit code $STATUS_EXIT_CODE. Aborting deployment." >&2
    exit $STATUS_EXIT_CODE
fi

# Use SSH with a heredoc for clarity and better command handling
# Pass the status exit code as an argument ($1) to the remote script
ssh "$REMOTE_HOST" bash -s -- "$STATUS_EXIT_CODE" << EOF
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

  # Retrieve the status code passed from the local script ($1 within the remote script)
  REMOTE_STATUS_CODE=\$1

  # Conditionally stop the server based on the status check performed *before* SSH
  if [ "\$REMOTE_STATUS_CODE" -eq 0 ]; then
    echo "[Remote] Status check indicated server was running. Attempting to stop..."
    # Find the PID again (necessary as it's a new remote session)
    PID=\$(pgrep -f "$SERVER_PATTERN" 2>/dev/null) || true
    if [ -n "\$PID" ]; then
        echo "[Remote] Found running server (PID: \$PID). Stopping..."
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
        # This case might happen if the server stopped between the status check and this command running
        echo "[Remote] Warning: Status check indicated running, but no process found now."
    fi
  else
    echo "[Remote] Status check indicated server was not running. Skipping stop step."
  fi

  # Ensure the status script itself is executable on the remote, in case it wasn't pulled/chmodded correctly before
  echo "[Remote] Ensuring status script is executable..."
  chmod +x ./get_status_of_beta1.sh || echo "[Remote] Warning: chmod failed for status script"

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
