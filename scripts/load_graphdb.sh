#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="http://localhost:7200/repositories/spb_1m/statements"
LOG_FILE="artifacts/logs/graphdb_loaded.txt"

mkdir -p "$(dirname "$LOG_FILE")"

load_file() {
  local file="$1"
  local ext="${file##*.}"
  local ct=""
  case "$ext" in
    ttl) ct="text/turtle" ;;
    nt) ct="application/n-triples" ;;
    nq) ct="application/n-quads" ;;
    rdf|xml) ct="application/rdf+xml" ;;
    *) return 0 ;;
  esac

  if grep -qx "$file" "$LOG_FILE" 2>/dev/null; then
    return 0
  fi

  echo "Loading $file..."
  curl -sS --retry 5 --retry-delay 3 --retry-all-errors \
    -H "Content-Type: $ct" \
    --data-binary "@$file" \
    "$ENDPOINT" >/dev/null
  echo "$file" >> "$LOG_FILE"
}

for target in "$@"; do
  if [[ -d "$target" ]]; then
    while IFS= read -r -d '' f; do
      load_file "$f"
      sleep 0.5
    done < <(find "$target" -type f \( -name '*.ttl' -o -name '*.nt' -o -name '*.nq' -o -name '*.rdf' -o -name '*.xml' \) \
      ! -path '*/data/sparql/*' ! -path '*/data/validation/*' -print0 | sort -z)
  elif [[ -f "$target" ]]; then
    load_file "$target"
  fi
 done
