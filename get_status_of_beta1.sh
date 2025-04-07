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
REMOTE_PORT="5000" # Port the Flask server runs on
HEALTH_ENDPOINT="http://localhost:$REMOTE_PORT/health"
# --- End Configuration ---

echo "--- Checking status of server on $REMOTE_HOST ---"

# Use SSH to execute commands remotely
# We capture the combined output to parse it locally
SSH_OUTPUT=$(ssh "$REMOTE_HOST" bash -s << EOF
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
  PID=\$(pgrep -f "$SERVER_PATTERN" 2>/dev/null) || true

  PROCESS_STATUS="NOT_RUNNING"
  HEALTH_STATUS="UNKNOWN"

  if [ -n "\$PID" ]; then
    PROCESS_STATUS="RUNNING"
    echo "[Remote] Server process is RUNNING (PID: \$PID)."

    echo "[Remote] Checking health endpoint: $HEALTH_ENDPOINT"
    # Use curl to check the health endpoint. Check HTTP status code.
    # -sS: Silent mode but show errors. -f: Fail silently on server errors (HTTP >= 400).
    # -o /dev/null: Discard response body. -w '%{http_code}': Output only the HTTP status code.
    # Set a timeout in case the server hangs.
    HTTP_CODE=\$(curl -sS -f -o /dev/null -w '%{http_code}' --max-time 5 "$HEALTH_ENDPOINT")
    CURL_EXIT_CODE=\$? # Capture curl's exit code

    if [ \$CURL_EXIT_CODE -eq 0 ] && [ "\$HTTP_CODE" -eq 200 ]; then
        echo "[Remote] Health check PASSED (HTTP 200)."
        HEALTH_STATUS="HEALTHY"
    elif [ \$CURL_EXIT_CODE -eq 22 ]; then # Curl error 22 often means 4xx/5xx with -f
        echo "[Remote] Health check FAILED (HTTP \$HTTP_CODE)."
        HEALTH_STATUS="UNHEALTHY"
    elif [ \$CURL_EXIT_CODE -ne 0 ]; then # Other curl errors (timeout, connection refused etc.)
        echo "[Remote] Health check FAILED (curl error \$CURL_EXIT_CODE)."
        HEALTH_STATUS="UNREACHABLE"
    else # Unexpected HTTP code (e.g., 3xx)
        echo "[Remote] Health check UNEXPECTED (HTTP \$HTTP_CODE)."
        HEALTH_STATUS="UNEXPECTED_STATUS"
    fi
  else
    echo "[Remote] Server process is NOT RUNNING."
    PROCESS_STATUS="NOT_RUNNING"
    HEALTH_STATUS="N/A" # Health is not applicable if process isn't running
  fi

  # Output status codes for local script parsing
  echo "PROCESS_STATUS=\$PROCESS_STATUS"
  echo "HEALTH_STATUS=\$HEALTH_STATUS"
EOF
)

# Check the exit status of the SSH command itself
SSH_EXIT_STATUS=$?
if [ $SSH_EXIT_STATUS -ne 0 ]; then
  echo "--- Error: SSH command failed with exit status $SSH_EXIT_STATUS ---" >&2
  echo "SSH Output:"
  echo "$SSH_OUTPUT"
  exit $SSH_EXIT_STATUS
fi

# Parse the output from the remote script
PROCESS_STATUS=$(echo "$SSH_OUTPUT" | grep 'PROCESS_STATUS=' | cut -d'=' -f2)
HEALTH_STATUS=$(echo "$SSH_OUTPUT" | grep 'HEALTH_STATUS=' | cut -d'=' -f2)

# Display the final status
echo "--- Server Status ---"
echo "Process: $PROCESS_STATUS"
echo "Health:  $HEALTH_STATUS"
echo "---------------------"

# Exit with 0 for success (status reported), 1 for error during check
exit 0
