#!/bin/bash
# filepath: /workspaces/kizlar-agha/scripts/generate-env.sh

set -e

EXAMPLE_ENV="${1:-.env.example.docker}"
GENERATED_ENV="$(echo "$EXAMPLE_ENV" | sed 's/example\.//')"

if [ ! -f "$EXAMPLE_ENV" ]; then
  echo "Error: $EXAMPLE_ENV not found!"
  exit 1
fi

# Generate a random password
PW=$(openssl rand -hex 32)

# Replace POSTGRES_PASSWORD line and write to .env
awk -v pw="$PW" '
  BEGIN { replaced=0 }
  /^POSTGRES_PASSWORD=/ {
    print "POSTGRES_PASSWORD=" pw
    replaced=1
    next
  }
  { print }
  END {
    if (!replaced) print "POSTGRES_PASSWORD=" pw
  }
' "$EXAMPLE_ENV" > "$GENERATED_ENV"

echo "Generated $GENERATED_ENV with random POSTGRES_PASSWORD:"
echo "$PW"