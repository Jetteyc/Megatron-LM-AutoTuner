#!/bin/bash

DOWNLOAD_LINK="$1"
MAX_RETRIES=9999
RETRY_INTERVAL=5  # 秒

if [ -z "$DOWNLOAD_LINK" ]; then
  echo "Usage: $0 <download-link>"
  exit 1
fi

attempt=1
while true; do
  echo "Attempt $attempt: downloading $DOWNLOAD_LINK ..."
  aria2c --max-tries=9999 "$DOWNLOAD_LINK"

  if [ $? -eq 0 ]; then
    echo "✅ Successfully downloaded $DOWNLOAD_LINK"
    break
  else
    echo "❌ Failed to download $DOWNLOAD_LINK"
    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
      echo "🚫 Reached maximum retry limit ($MAX_RETRIES). Exiting."
      exit 1
    fi
    attempt=$((attempt + 1))
    echo "⏳ Retrying in $RETRY_INTERVAL seconds..."
    sleep $RETRY_INTERVAL
  fi
done
