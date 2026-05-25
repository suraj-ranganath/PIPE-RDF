#!/usr/bin/env bash
set -euo pipefail

GRAPHDB_VERSION="${GRAPHDB_VERSION:-10.6.3}"
GRAPHDB_ROOT="${GRAPHDB_ROOT:-/data/suraj/graphdb-user}"
GRAPHDB_INSTALL_DIR="${GRAPHDB_INSTALL_DIR:-$GRAPHDB_ROOT/graphdb-$GRAPHDB_VERSION}"
GRAPHDB_DOWNLOAD_DIR="${GRAPHDB_DOWNLOAD_DIR:-$GRAPHDB_ROOT/downloads}"
GRAPHDB_DIST_URL="${GRAPHDB_DIST_URL:-https://maven.ontotext.com/repository/owlim-releases/com/ontotext/graphdb/graphdb/$GRAPHDB_VERSION/graphdb-$GRAPHDB_VERSION-dist.zip}"
GRAPHDB_DIST_MD5="${GRAPHDB_DIST_MD5:-dbb4a784229ad3610ee0dd46aeddb523}"
GRAPHDB_HOME="${GRAPHDB_HOME:-$GRAPHDB_ROOT/home}"
GRAPHDB_DATA_DIR="${GRAPHDB_DATA_DIR:-$GRAPHDB_HOME/data}"
GRAPHDB_BACKUP_SRC="${GRAPHDB_BACKUP_SRC:-/data/suraj/pipe-rdf-arr/graphdb_data_backup}"
GRAPHDB_PORT="${GRAPHDB_PORT:-7200}"
GRAPHDB_HEAP="${GRAPHDB_HEAP:-64g}"
GRAPHDB_LOG_DIR="${GRAPHDB_LOG_DIR:-$GRAPHDB_ROOT/logs}"
GRAPHDB_PID_FILE="${GRAPHDB_PID_FILE:-$GRAPHDB_ROOT/graphdb.pid}"
CONDA_ENV="${CONDA_ENV:-pipe-rdf-arr}"

REPOS=("spb_1m" "spb_company_mini_slice")

mkdir -p "$GRAPHDB_ROOT" "$GRAPHDB_DOWNLOAD_DIR" "$GRAPHDB_LOG_DIR"

activate_conda_env() {
  if [ -n "${CONDA_PREFIX:-}" ] && [ "$(basename "$CONDA_PREFIX")" = "$CONDA_ENV" ]; then
    return
  fi
  if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" >/dev/null
  elif [ -f "$HOME/miniforge3/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/bin/activate" "$CONDA_ENV" >/dev/null
  fi
}

java_home_dir() {
  activate_conda_env
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/java" ]; then
    printf "%s\n" "$CONDA_PREFIX"
    return
  fi
  if [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/java" ]; then
    printf "%s\n" "$JAVA_HOME"
    return
  fi
  if command -v java >/dev/null 2>&1; then
    dirname "$(dirname "$(readlink -f "$(command -v java)")")"
    return
  fi
  echo "Could not find Java. Install openjdk=17 in conda env $CONDA_ENV or set JAVA_HOME." >&2
  return 1
}

dist_zip_path() {
  printf "%s/graphdb-%s-dist.zip" "$GRAPHDB_DOWNLOAD_DIR" "$GRAPHDB_VERSION"
}

verify_dist_zip() {
  local zip_path="$1"
  [ -s "$zip_path" ] || return 1
  if command -v md5sum >/dev/null 2>&1; then
    printf "%s  %s\n" "$GRAPHDB_DIST_MD5" "$zip_path" | md5sum -c --status -
  else
    return 0
  fi
}

download_graphdb_dist() {
  if [ -x "$GRAPHDB_INSTALL_DIR/bin/graphdb" ]; then
    return
  fi

  local zip_path
  zip_path="$(dist_zip_path)"
  if ! verify_dist_zip "$zip_path"; then
    rm -f "$zip_path"
    echo "Downloading GraphDB $GRAPHDB_VERSION distribution"
    curl -fL --retry 8 --retry-delay 5 --connect-timeout 30 \
      -o "$zip_path" "$GRAPHDB_DIST_URL"
  fi
  verify_dist_zip "$zip_path"

  echo "Extracting GraphDB into $GRAPHDB_ROOT"
  rm -rf "$GRAPHDB_INSTALL_DIR"
  python3 - "$zip_path" "$GRAPHDB_ROOT" <<'PY'
import os
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1]).resolve()
out_root = Path(sys.argv[2]).resolve()
with zipfile.ZipFile(zip_path) as zf:
    for member in zf.infolist():
        target = (out_root / member.filename).resolve()
        if os.path.commonpath([str(out_root), str(target)]) != str(out_root):
            raise SystemExit(f"Refusing unsafe zip member: {member.filename}")
    zf.extractall(out_root)
PY
  chmod +x "$GRAPHDB_INSTALL_DIR/bin/"* 2>/dev/null || true
  if [ ! -x "$GRAPHDB_INSTALL_DIR/bin/graphdb" ]; then
    echo "GraphDB executable was not found at $GRAPHDB_INSTALL_DIR/bin/graphdb" >&2
    find "$GRAPHDB_ROOT" -maxdepth 3 -type f -name graphdb -print >&2 || true
    return 1
  fi
}

restore_backup() {
  if [ ! -d "$GRAPHDB_BACKUP_SRC/repositories" ]; then
    echo "Missing GraphDB backup source: $GRAPHDB_BACKUP_SRC" >&2
    return 1
  fi
  echo "Restoring GraphDB data from $GRAPHDB_BACKUP_SRC"
  mkdir -p "$GRAPHDB_DATA_DIR"
  rsync -a --delete "$GRAPHDB_BACKUP_SRC/" "$GRAPHDB_DATA_DIR/"
  rm -rf "$GRAPHDB_HOME/repositories"
  rm -rf "$GRAPHDB_DATA_DIR/repositories/lock" "$GRAPHDB_DATA_DIR/repositories"/*/storage/lock
  find "$GRAPHDB_DATA_DIR/repositories" -type f -name "*.lock" -delete 2>/dev/null || true
}

is_running() {
  [ -f "$GRAPHDB_PID_FILE" ] && kill -0 "$(cat "$GRAPHDB_PID_FILE")" >/dev/null 2>&1
}

rest_api_healthy() {
  curl -fsS --max-time 5 "http://127.0.0.1:$GRAPHDB_PORT/rest/repositories" >/dev/null
}

stop_graphdb() {
  local pid
  if is_running; then
    pid="$(cat "$GRAPHDB_PID_FILE")"
    echo "Stopping GraphDB pid $pid"
    kill "$pid" >/dev/null 2>&1 || true
  fi

  if command -v pgrep >/dev/null 2>&1; then
    while read -r pid; do
      [ -n "$pid" ] || continue
      [ "$pid" = "$$" ] && continue
      echo "Stopping GraphDB process $pid"
      kill "$pid" >/dev/null 2>&1 || true
    done < <(pgrep -f "graphdb.home=$GRAPHDB_HOME" || true)
  fi

  for _ in $(seq 1 45); do
    if ! is_running && ! rest_api_healthy; then
      rm -f "$GRAPHDB_PID_FILE"
      return
    fi
    sleep 1
  done

  if command -v pgrep >/dev/null 2>&1; then
    while read -r pid; do
      [ -n "$pid" ] || continue
      [ "$pid" = "$$" ] && continue
      echo "Force-stopping GraphDB process $pid"
      kill -9 "$pid" >/dev/null 2>&1 || true
    done < <(pgrep -f "graphdb.home=$GRAPHDB_HOME" || true)
  fi
  rm -f "$GRAPHDB_PID_FILE"
}

start_graphdb() {
  download_graphdb_dist
  if [ ! -d "$GRAPHDB_DATA_DIR/repositories/spb_1m" ] || [ ! -d "$GRAPHDB_DATA_DIR/repositories/spb_company_mini_slice" ]; then
    restore_backup
  fi
  if is_running; then
    echo "GraphDB already running with pid $(cat "$GRAPHDB_PID_FILE")"
    return
  fi
  if rest_api_healthy; then
    echo "GraphDB already responds on port $GRAPHDB_PORT"
    return
  fi

  local java_home log_file
  java_home="$(java_home_dir)"
  log_file="$GRAPHDB_LOG_DIR/graphdb.log"
  : > "$log_file"
  echo "Starting GraphDB on port $GRAPHDB_PORT with home $GRAPHDB_HOME"
  GRAPHDB_HOME="$GRAPHDB_HOME" \
  JAVA_HOME="$java_home" \
  GDB_HEAP_SIZE="$GRAPHDB_HEAP" \
  PATH="$java_home/bin:$GRAPHDB_INSTALL_DIR/bin:$PATH" \
  nohup "$GRAPHDB_INSTALL_DIR/bin/graphdb" \
    "-Dgraphdb.home=$GRAPHDB_HOME" \
    "-Dgraphdb.connector.port=$GRAPHDB_PORT" \
    > "$log_file" 2>&1 &
  echo $! > "$GRAPHDB_PID_FILE"
}

wait_graphdb() {
  for _ in $(seq 1 180); do
    if rest_api_healthy; then
      return 0
    fi
    if [ -f "$GRAPHDB_PID_FILE" ] && ! kill -0 "$(cat "$GRAPHDB_PID_FILE")" >/dev/null 2>&1; then
      echo "GraphDB exited before becoming healthy" >&2
      tail -n 100 "$GRAPHDB_LOG_DIR/graphdb.log" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "GraphDB did not become healthy on port $GRAPHDB_PORT" >&2
  tail -n 100 "$GRAPHDB_LOG_DIR/graphdb.log" >&2 || true
  return 1
}

query_count() {
  local repo="$1"
  curl -fsS --max-time 120 -G \
    --data-urlencode 'query=SELECT (COUNT(*) AS ?triples) WHERE { ?s ?p ?o }' \
    -H 'Accept: application/sparql-results+json' \
    "http://127.0.0.1:$GRAPHDB_PORT/repositories/$repo" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["results"]["bindings"][0]["triples"]["value"])'
}

health() {
  rest_api_healthy
  for repo in "${REPOS[@]}"; do
    printf "%s triples: " "$repo"
    query_count "$repo"
  done
}

case "${1:-status}" in
  setup)
    download_graphdb_dist
    restore_backup
    start_graphdb
    wait_graphdb
    health
    ;;
  start)
    start_graphdb
    wait_graphdb
    health
    ;;
  stop)
    stop_graphdb
    ;;
  restart)
    stop_graphdb
    start_graphdb
    wait_graphdb
    health
    ;;
  health|status)
    if ! health; then
      echo "GraphDB health check failed; restarting once" >&2
      stop_graphdb
      start_graphdb
      wait_graphdb
      health
    fi
    ;;
  restore)
    stop_graphdb
    restore_backup
    ;;
  *)
    echo "Usage: $0 {setup|start|stop|restart|health|status|restore}" >&2
    exit 2
    ;;
esac
