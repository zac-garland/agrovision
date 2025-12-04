#!/bin/bash
# Quick script to test an image file via the /diagnose endpoint

IMAGE_FILE="${1:-static/test-image2.jpeg}"
ENDPOINT="${2:-http://127.0.0.1:5000/diagnose}"

if [ ! -f "$IMAGE_FILE" ]; then
    echo "❌ Image file not found: $IMAGE_FILE"
    exit 1
fi

echo "🔍 Testing image: $IMAGE_FILE"
echo "📡 Endpoint: $ENDPOINT"
echo ""

# Test using curl
curl -X POST \
  -F "image=@$IMAGE_FILE" \
  "$ENDPOINT" | python3 -m json.tool

