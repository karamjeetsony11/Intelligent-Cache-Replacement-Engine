set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_PYTHON=false
if [[ "${1:-}" == "--python" ]]; then
    RUN_PYTHON=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--python]"
    exit 2
fi

echo "========================================"
echo "   INTELLIGENT CACHE - FULL RUN"
echo "========================================"

echo
echo "[1/5] Checking native build environment..."

command -v g++ >/dev/null 2>&1 || {
    echo "ERROR: g++ is not installed."
    exit 1
}

command -v make >/dev/null 2>&1 || {
    echo "ERROR: make is not installed."
    exit 1
}

echo "✓ g++ found"
echo "✓ make found"


echo
echo "[2/5] Building C++ cache engine..."

make

echo "✓ C++ engine built"


echo
echo "[3/5] Generating workload + Bélády labels..."

mkdir -p data models

./cache_engine generate \
    --requests 100000 \
    --items 1000 \
    --capacity 100 \
    --output data/labeled_requests.csv

echo "✓ Dataset generated:"
echo "  data/labeled_requests.csv"

echo
echo "[4/5] Running C++ cache benchmark..."

./cache_engine benchmark \
    --requests 100000 \
    --items 1000 \
    --capacity 100 \
    --skew 1.2 \
    --seed 42

echo "✓ C++ benchmark completed"


echo
echo "[5/5] Running C++ self-tests..."

make test

echo "✓ Self-tests completed"


if [[ "$RUN_PYTHON" == true ]]; then
    echo
    echo "----- Optional Python offline experiments -----"
    command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is not installed."; exit 1; }
    python3 -m pip install -r requirements.txt
    python3 model.py
    python3 test_cache.py
    python3 benchmark.py
    python3 test_online_cache.py
fi

echo
echo "========================================"
echo "        ALL EXPERIMENTS COMPLETED"
echo "========================================"

echo
echo "Generated files:"
echo "  data/labeled_requests.csv"
if [[ "$RUN_PYTHON" == true ]]; then
    echo "  models/best_model.pkl"
fi

echo
echo "Done."
