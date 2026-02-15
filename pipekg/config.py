from pathlib import Path

# Reproducibility
SEED = 42

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
FIG_DIR = ARTIFACTS_DIR / "figures"
LOG_DIR = ARTIFACTS_DIR / "logs"

# Synthetic KG sizes
NUM_FILMS = 200
NUM_DIRECTORS = 35
NUM_ACTORS = 90
NUM_PRODUCERS = 25
NUM_GENRES = 12
NUM_COUNTRIES = 10

# Dataset sizes
PHASE1_SEEDS_PER_TEMPLATE = 8
PHASE1_TEMPLATES_PER_CATEGORY = 5
PHASE2_SEEDS_PER_CATEGORY = 20
PHASE3_SAMPLES_PER_CATEGORY = 40

# Real experiment sizes
TARGET_PER_CATEGORY = 50

# Repair attempts
REPAIR_ATTEMPTS = 1

# SPARQL endpoint (GraphDB or other)
SPARQL_ENDPOINT_URL = "http://localhost:7200/repositories/spb_1m"

# Embedding model defaults (to be set when key is provided)
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

# Duplicate detection
DUP_SIM_THRESHOLD = 0.95

# Retrieval
RETRIEVAL_TOP_K = 3

# Benchmark categories (Mintaka-style + generic)
CATEGORIES = [
    "generic",
    "counting",
    "comparative",
    "superlative",
    "ordinal",
    "multi-hop",
    "intersection",
    "difference",
    "yesno",
]

# Prediction corruption for evaluation
CORRUPTION_RATE = 0.3
