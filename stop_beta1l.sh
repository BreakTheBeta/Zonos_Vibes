#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
REMOTE_HOST="beta1l"
# --- End Configuration ---

echo "--- Attempting to stop server processes on $REMOTE_HOST ---"

# Use SSH with a heredoc
ssh "$REMOTE_HOST" bash -s << EOF
  # Define patterns *inside* the remote script block
  # Ensure backslashes are escaped for the remote shell AND for pgrep/pkill regex
  REMOTE_SPECIFIC_PATTERN='\\.venv/bin/python3 server\\.py'
  REMOTE_UV_PATTERN='uv run server\\.py'

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

  echo "[Remote] Attempting to stop processes matching '\$REMOTE_SPECIFIC_PATTERN' or '\$REMOTE_UV_PATTERN'..."
  # Use pkill -f with the locally defined remote patterns. Send SIGTERM first, then SIGKILL.
  # The || true prevents the script from exiting if pkill finds no processes.
  pkill -f "\$REMOTE_SPECIFIC_PATTERN" || true
  pkill -f "\$REMOTE_UV_PATTERN" || true
  sleep 1 # Give processes a moment to terminate

  # Check if any are still running and send SIGKILL
  if pgrep -f "\$REMOTE_SPECIFIC_PATTERN" > /dev/null || pgrep -f "\$REMOTE_UV_PATTERN" > /dev/null; then
      echo "[Remote] Some server processes still running, sending SIGKILL..."
      pkill -9 -f "\$REMOTE_SPECIFIC_PATTERN" || true
      pkill -9 -f "\$REMOTE_UV_PATTERN" || true
      sleep 1
  fi

  # Final verification
  if pgrep -f "\$REMOTE_SPECIFIC_PATTERN" > /dev/null || pgrep -f "\$REMOTE_UV_PATTERN" > /dev/null; then
      echo "[Remote] Warning: Server processes might still be running after kill attempts."
      # Optionally exit with an error code here if needed
      # exit 1
  else
      echo "[Remote] Server processes stopped successfully."
  fi

  echo "[Remote] Stop script finished."
EOF

# Check the exit status of the SSH command itself
SSH_EXIT_STATUS=$?
if [ $SSH_EXIT_STATUS -ne 0 ]; then
  echo "--- Error: SSH command failed with exit status $SSH_EXIT_STATUS ---" >&2
  exit $SSH_EXIT_STATUS
fi

echo "--- Stop script finished successfully ---"
exit 0
