#!/usr/bin/env bash
# setup_workspace.sh
# Creates the runtime workspace for the FedRAG system.
# Run once before starting services: bash setup_workspace.sh
# Safe to re-run (idempotent).

set -euo pipefail

WORKSPACE="${1:-./workspace}"
echo "Setting up FedRAG workspace at: $WORKSPACE"

# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------

mkdir -p "$WORKSPACE/coordinator"
mkdir -p "$WORKSPACE/chroma/node-1"
mkdir -p "$WORKSPACE/chroma/node-2"
mkdir -p "$WORKSPACE/chroma/node-3"
mkdir -p "$WORKSPACE/data/node-1"
mkdir -p "$WORKSPACE/data/node-2"
mkdir -p "$WORKSPACE/data/node-3"
mkdir -p "$WORKSPACE/logs"

echo "  Created directory structure"

# ---------------------------------------------------------------------------
# Copy sample documents into each node's data folder
# (skips if already present)
# ---------------------------------------------------------------------------

SRC="nodes/data"

copy_if_missing() {
    local src="$1"
    local dst="$2"
    if [ ! -f "$dst/$(basename $src)" ]; then
        cp "$src" "$dst/"
        echo "  Copied $(basename $src) → $dst/"
    else
        echo "  Skipped $(basename $src) (already exists)"
    fi
}

if [ -f "$SRC/node_finance/finance_basics.md" ]; then
    copy_if_missing "$SRC/node_finance/finance_basics.md" "$WORKSPACE/data/node-1"
fi
if [ -f "$SRC/node_medical/medical_basics.md" ]; then
    copy_if_missing "$SRC/node_medical/medical_basics.md" "$WORKSPACE/data/node-2"
fi
if [ -f "$SRC/node_legal/legal_basics.md" ]; then
    copy_if_missing "$SRC/node_legal/legal_basics.md" "$WORKSPACE/data/node-3"
fi

echo "  Sample documents ready"

# ---------------------------------------------------------------------------
# Generate per-node config files
# ---------------------------------------------------------------------------

write_node_config() {
    local node_id="$1"
    local domain="$2"
    local out="$WORKSPACE/config_${node_id//-/_}.yaml"

    cat > "$out" <<YAML
node_id: "$node_id"
domain: "$domain"
server_address: "coordinator:9091"
persist_directory: "/workspace/chroma/$node_id"
collection_name: "fedrag_${node_id//-/_}"
data_directory: "/workspace/data/$node_id"
embedding_model_name: "all-MiniLM-L6-v2"
embedding_dimension: 384
top_k_default: 5
YAML
    echo "  Written $out"
}

write_node_config "node-1" "finance"
write_node_config "node-2" "medical"
write_node_config "node-3" "legal"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "Workspace ready at: $WORKSPACE"
echo ""
echo "Contents:"
find "$WORKSPACE" -type f | sort | sed 's/^/  /'
echo ""
echo "Next steps:"
echo "  docker compose up --build"
