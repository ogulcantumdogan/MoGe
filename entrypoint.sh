#!/usr/bin/env bash
set -euo pipefail

# Auto-shutdown after specified time
# Default: 3 hours (10800 seconds)
SHUTDOWN_TIMEOUT=${SHUTDOWN_TIMEOUT:-10800}
echo "🕒 Auto-shutdown timer set for $(($SHUTDOWN_TIMEOUT / 60)) minutes"

# Kick off a background timer that will tell Vast.ai to stop this instance
(
  # Wait for the specified timeout
  sleep $SHUTDOWN_TIMEOUT
  
  # Get instance ID from vast.ai environment
  INSTANCE_ID=$(cat /etc/vast_instance_id 2>/dev/null || echo "")
  
  if [ -n "$INSTANCE_ID" ]; then
    echo "⚠️ Auto-shutdown triggered after $(($SHUTDOWN_TIMEOUT / 60)) minutes of runtime"
    echo "🛑 Stopping instance $INSTANCE_ID"
    
    # Use vast-ai API to stop the instance
    if [ -n "$VAST_API_KEY" ]; then
      curl -s -X PUT "https://console.vast.ai/api/v0/instances/$INSTANCE_ID/" \
        -H "Accept: application/json" \
        -H "Authorization: Bearer $VAST_API_KEY" \
        -d '{"state": "stopped"}'
      echo "✅ Shutdown request sent successfully"
    else
      echo "❌ VAST_API_KEY not set. Cannot auto-shutdown."
    fi
  else
    echo "❓ Could not determine instance ID. Auto-shutdown failed."
  fi
) &

# Now run the command that was passed to the entrypoint
echo "🚀 Starting MoGe application..."
exec "$@"
