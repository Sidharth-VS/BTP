#!/usr/bin/env bash
# setup_workspace.sh
# Creates the runtime workspace for the FedRAG system from nodes/nodes.yaml.
# Run once before starting services: bash setup_workspace.sh [workspace_dir]
# Safe to re-run (idempotent).

set -euo pipefail

WORKSPACE="${1:-./workspace}"
NODES_YAML="nodes/nodes.yaml"

if [ ! -f "$NODES_YAML" ]; then
    echo "ERROR: $NODES_YAML not found. Run from the project root."
    exit 1
fi

echo "Setting up FedRAG workspace at: $WORKSPACE"
echo "Reading node registry from: $NODES_YAML"
echo ""

# Parse node IDs and domains from nodes.yaml using Python (already in venv)
PYTHON="${PYTHON:-python}"
NODE_LIST=$($PYTHON - <<'PYEOF'
import yaml, sys
with open("nodes/nodes.yaml") as f:
    data = yaml.safe_load(f)
for node in data.get("nodes", []):
    print(f"{node['node_id']}:{node.get('domain','general')}")
PYEOF
)

# ---------------------------------------------------------------------------
# Create coordinator workspace dir
# ---------------------------------------------------------------------------
mkdir -p "$WORKSPACE/coordinator"
echo "  [coordinator] workspace/coordinator/"

# ---------------------------------------------------------------------------
# Create per-node directories and copy sample data
# ---------------------------------------------------------------------------
while IFS=: read -r NODE_ID DOMAIN; do
    echo "  [${NODE_ID}] domain=${DOMAIN}"

    mkdir -p "$WORKSPACE/chroma/${NODE_ID}"
    mkdir -p "$WORKSPACE/data/${NODE_ID}"

    # Copy sample documents if they exist in nodes/data/<node_id>/
    SRC_DIR="nodes/data/${NODE_ID}"
    DST_DIR="$WORKSPACE/data/${NODE_ID}"
    if [ -d "$SRC_DIR" ]; then
        for f in "$SRC_DIR"/*; do
            [ -f "$f" ] || continue
            FNAME=$(basename "$f")
            if [ ! -f "$DST_DIR/$FNAME" ]; then
                cp "$f" "$DST_DIR/"
                echo "    Copied $FNAME → workspace/data/${NODE_ID}/"
            else
                echo "    Skipped $FNAME (already present)"
            fi
        done
    else
        echo "    WARNING: No sample data found at $SRC_DIR"
        echo "             Place your documents there and re-run setup_workspace.sh"
    fi

done <<< "$NODE_LIST"

# ---------------------------------------------------------------------------
# Create logs dir
# ---------------------------------------------------------------------------
mkdir -p "$WORKSPACE/logs"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "Workspace ready at: $WORKSPACE"
echo ""
echo "Contents:"
find "$WORKSPACE" -type f | sort | sed 's/^/  /'
echo ""
echo "To add a new node:"
echo "  1. Append an entry to nodes/nodes.yaml"
echo "  2. Place documents in nodes/data/<new_node_id>/"
echo "  3. Re-run: bash setup_workspace.sh"
echo "  4. Add a service to docker-compose.yml"
echo ""
echo "Next step: docker compose up --build"
