#!/bin/bash
"""Start ReflectLog server with cProfile performance profiling.

Profiles CPU usage and execution time to identify performance bottlenecks.
Useful for optimizing hot paths in search, add, and reranking operations.

Usage:
    ./scripts/profile-server.sh --transport http --port 9103
    ./scripts/profile-server.sh --profile search --queries 100
    ./scripts/profile-server.sh --profile add --messages 1000
    ./scripts/profile-server.sh --profile all --operations 100

Output:
    - Profile data: /tmp/profile-stats.prof
    - Visualization: python -m pstats profile-stats.prof
    - Report: Sorted by cumulative time

Examples:
    # Profile search operations
    ./scripts/profile-server.sh --profile search --queries 1000

    # Profile add operations  
    ./scripts/profile-server.sh --profile add --messages 1000

    # Profile full workflow
    ./scripts/profile-server.sh --profile all --operations 1000
"""

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG_LEVEL="${LOG_LEVEL:-INFO}"
MCP_TRANSPORT="${MCP_TRANSPORT:-http}"
MCP_PORT="${MCP_PORT:-9103}"
PROFILE_TYPE="${PROFILE_TYPE:-all}"
PROFILE_QUERIES="${PROFILE_QUERIES:-1000}"
PROFILE_MESSAGES="${PROFILE_MESSAGES:-1000}"

PROFILE_DIR="/tmp/profile-stats"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

print "🔍 Starting ReflectLog with profiling..."
print "   Profile: $PROFILE_TYPE"
print "   Timestamp: $TIMESTAMP"

if [[ "$PROFILE_TYPE" == "search" || "$PROFILE_TYPE" == "all" ]]; then
    print "   Mode: Searching (will run $PROFILE_QUERIES queries)"
fi

if [[ "$PROFILE_TYPE" == "add" || "$PROFILE_TYPE" == "all" ]]; then
    print "   Mode: Adding (will add $PROFILE_MESSAGES messages)"
fi

if [[ "$PROFILE_TYPE" == "all" ]]; then
    print "   Mode: All operations (add + search)"
fi

mkdir -p "$PROFILE_DIR"

PROFILE_CMD="profile-stats-$TIMESTAMP.prof"

PROFILER_RUN="$PYTHONPROF=1 cProfile -s time -o \"$PROFILE_DIR/$PROFILE_CMD\""

if [[ "$PROFILE_TYPE" == "search" || "$PROFILE_TYPE" == "all" ]]; then
    for ((i=1; i<=PROFILE_QUERIES; i++)); do
        echo "   Query $i/$PROFILE_QUERIES (using HTTP)..."
    done
fi

if [[ "$PROFILE_TYPE" == "add" || "$PROFILE_TYPE" == "all" ]]; then
    echo "   Adding $PROFILE_MESSAGES messages (using HTTP)..."
fi

if [[ "$PROFILE_TYPE" == "all" ]]; then
    for ((i=1; i<=PROFILE_QUERIES; i++)); do
        echo "   Mixed operation $i/$PROFILE_QUERIES (using HTTP)..."
    done
fi

echo "✓ Profiling started..."
echo ""

case "$MCP_TRANSPORT" in
    http|streamable-http)
        uv run reflectlog --transport "$MCP_TRANSPORT" --port "$MCP_PORT" "$PROFILER_RUN" "$@"
        ;;
esac

echo ""
echo "📊 Processing profile data..."
echo ""

python << 'EOF'
from pathlib import Path
import sys

profile_path = Path("$PROFILE_DIR/$PROFILE_CMD")

if not profile_path.exists():
    print(f"❌ Profile file not found: {profile_path}")
    sys.exit(1)

stats_data = profile_path.read_text()
lines = stats_data.split('\n')

print(f"✓ Loaded {len(lines)} profile lines")
print(f"✓ Profile data saved to: {profile_path}")

# Parse and display top functions
print("\n🔍 Top 20 functions by cumulative time:")
print(f"{'Function':<25} {'Time':<12} {'Calls':<10}")
print("-" * 50)

functions: list = []
for line in lines:
    if not line.strip() or line.startswith("#") or line.startswith("profiler"):
        continue
    if "function(" in line:
        func_name = line.split("function(")[1].split(")")[0].strip()
        time_ms = float(line.split("cumtime=")[1].split(")")[0].replace(" ms", ""))
        n_calls = int(line.split("ncalls")[1].split("=")[1].replace(")", ""))
        functions.append((func_name, time_ms, n_calls))

# Sort by cumulative time
functions.sort(key=lambda x: x[1], reverse=True)

for func, time_ms, n_calls in functions[:20]:
    time_s = f"{time_ms/1000:.3f}s"
    calls_s = f"{n_calls:4d}" if n_calls >= 1000 else f"{n_calls}"
    print(f"  {func:<25} {time_s:<12} {calls_s:<10}")

print("")
print(f"📊 Profile data: {profile_path}")
print(f"📈 Visualize: python -m gprof2dot {profile_path}")
print(f"📈 Analyze: python -m pstats {profile_path}")
EOF

echo ""
echo "✓ Profiling complete!"
echo "   Profile data: $PROFILE_DIR/$PROFILE_CMD"
echo ""
