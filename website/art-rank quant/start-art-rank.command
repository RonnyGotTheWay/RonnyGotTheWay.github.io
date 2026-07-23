#!/bin/zsh

set -u

project_root="${0:A:h}"
cd "$project_root"

# Remove only ART-RANK jobs left by an interrupted automated startup.
launchctl remove com.art-rank.web >/dev/null 2>&1 || true
launchctl remove com.art-rank.api >/dev/null 2>&1 || true

web_pid=""
api_pid=""

cleanup() {
  if [[ -n "$web_pid" ]]; then
    kill "$web_pid" >/dev/null 2>&1 || true
  fi
  if [[ -n "$api_pid" ]]; then
    kill "$api_pid" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

print "ART-RANK supervisor is active."
print "Web: http://127.0.0.1:3000"
print "API: http://127.0.0.1:8000"
print "Keep this Terminal window open. Press Control-C to stop ART-RANK."

while true; do
  if ! lsof -nP -iTCP:3000 -sTCP:LISTEN >/dev/null 2>&1; then
    print "Starting ART-RANK Web..."
    npm run dev:web &
    web_pid=$!
  fi

  if ! lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
    print "Starting ART-RANK API..."
    make api &
    api_pid=$!
  fi

  sleep 5
done
