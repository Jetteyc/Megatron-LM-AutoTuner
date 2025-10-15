#!/bin/bash

IMAGE_NAME="$1"
MAX_RETRIES=9999
RETRY_INTERVAL=5  # 秒

if [ -z "$IMAGE_NAME" ]; then
  echo "Usage: $0 <image-name>"
  exit 1
fi

attempt=1
while true; do
  echo "Attempt $attempt: pulling $IMAGE_NAME ..."
  docker pull "$IMAGE_NAME"

  if [ $? -eq 0 ]; then
    echo "✅ Successfully pulled $IMAGE_NAME"
    break
  else
    echo "❌ Failed to pull $IMAGE_NAME"
    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
      echo "🚫 Reached maximum retry limit ($MAX_RETRIES). Exiting."
      exit 1
    fi
    attempt=$((attempt + 1))
    echo "⏳ Retrying in $RETRY_INTERVAL seconds..."
    sleep $RETRY_INTERVAL
  fi
done
