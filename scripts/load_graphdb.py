import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

CONTENT_TYPES = {
    ".ttl": "text/turtle",
    ".nt": "application/n-triples",
    ".nq": "application/n-quads",
    ".rdf": "application/rdf+xml",
    ".xml": "application/rdf+xml",
}


LOG_PATH = Path("artifacts/logs/graphdb_loaded.txt")


def load_log() -> set[str]:
    if LOG_PATH.exists():
        return set(line.strip() for line in LOG_PATH.read_text().splitlines() if line.strip())
    return set()


def append_log(path: Path) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(str(path) + "\n")


def upload_file(endpoint: str, path: Path, retries: int = 3) -> None:
    ext = path.suffix.lower()
    content_type = CONTENT_TYPES.get(ext)
    if content_type is None:
        return
    data = path.read_bytes()
    for attempt in range(1, retries + 1):
        try:
            req = Request(endpoint, data=data, method="POST")
            req.add_header("Content-Type", content_type)
            req.add_header("Connection", "close")
            with urlopen(req) as resp:
                if resp.status not in (200, 201, 204):
                    raise RuntimeError(f"Upload failed for {path.name}: {resp.status}")
            return
        except (URLError, RuntimeError) as exc:
            if attempt >= retries:
                raise
            time.sleep(2 * attempt)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: load_graphdb.py <endpoint> <dir_or_file> [...]")
        sys.exit(1)
    endpoint = sys.argv[1]
    loaded = load_log()
    for target in sys.argv[2:]:
        path = Path(target)
        if path.is_dir():
            for file in sorted(path.glob("**/*")):
                if file.is_file() and file.suffix.lower() in CONTENT_TYPES:
                    if str(file) in loaded:
                        continue
                    print(f"Loading {file}...")
                    upload_file(endpoint, file)
                    append_log(file)
                    time.sleep(0.5)
        elif path.is_file():
            if str(path) in loaded:
                continue
            print(f"Loading {path}...")
            upload_file(endpoint, path)
            append_log(path)


if __name__ == "__main__":
    main()
