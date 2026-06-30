#!/bin/sh
set -eu

api_base_url="${SIGNALFORGE_API_BASE_URL:-http://localhost:8000}"

escaped_api_base_url=$(printf '%s' "$api_base_url" | sed 's/\\/\\\\/g; s/"/\\"/g')

cat > /usr/share/nginx/html/runtime-config.js <<EOF
window.SIGNALFORGE_RUNTIME_CONFIG = {
  apiBaseUrl: "$escaped_api_base_url"
};
EOF
