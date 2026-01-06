#!/bin/bash
# Redbank Demo Deployment Script
#
# This script deploys the Redbank agentic demo using:
# - PostgreSQL and MCP server from rag/demos/redbank-demo/
# - AgentWorkflow from this example directory
#
# Usage: ./deploy.sh [NAMESPACE]
#   NAMESPACE: Target namespace (default: redbank-agentic)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REDBANK_DEMO_DIR="${REPO_ROOT}/rag/demos/redbank-demo"
NAMESPACE="${1:-redbank-agentic}"

# Detect kubectl/oc
KUBECTL="kubectl"
if command -v oc &> /dev/null; then
    KUBECTL="oc"
fi

echo "=============================================="
echo "Redbank Agentic Demo Deployment"
echo "=============================================="
echo "Namespace: ${NAMESPACE}"
echo "Redbank demo source: ${REDBANK_DEMO_DIR}"
echo "Using: ${KUBECTL}"
echo "=============================================="

# Verify source directory exists
if [[ ! -d "${REDBANK_DEMO_DIR}" ]]; then
    echo "ERROR: Redbank demo directory not found at ${REDBANK_DEMO_DIR}"
    echo "Make sure you're running this from the agentic-flow monorepo"
    exit 1
fi

# Create namespace
echo ""
echo "[1/6] Creating namespace ${NAMESPACE}..."
${KUBECTL} create namespace "${NAMESPACE}" 2>/dev/null || echo "  Namespace already exists"

# Allow image pulls from operator namespace
echo ""
echo "[2/6] Setting up image pull permissions..."
${KUBECTL} policy add-role-to-user system:image-puller \
    "system:serviceaccount:${NAMESPACE}:default" \
    -n rh-agentic-system 2>/dev/null || echo "  Permission already granted or oc not available"

# Label namespace for Kagenti discovery
echo ""
echo "[3/6] Labeling namespace for Kagenti..."
${KUBECTL} label namespace "${NAMESPACE}" kagenti-enabled=true --overwrite

# Deploy PostgreSQL
echo ""
echo "[4/6] Deploying PostgreSQL with seed data..."
${KUBECTL} apply -k "${REDBANK_DEMO_DIR}/postgres-db/" -n "${NAMESPACE}"
echo "  Waiting for PostgreSQL to be ready..."
${KUBECTL} wait --for=condition=available --timeout=120s deployment/postgresql -n "${NAMESPACE}" || {
    echo "  WARNING: PostgreSQL not ready yet, continuing..."
}

# Build and deploy MCP server
echo ""
echo "[5/6] Building and deploying Redbank MCP server..."
cd "${REDBANK_DEMO_DIR}/mcp-server"

# Create BuildConfig if it doesn't exist
${KUBECTL} get buildconfig build-redbank-mcp-server -n "${NAMESPACE}" &>/dev/null || {
    echo "  Creating BuildConfig..."
    ${KUBECTL} new-build --name build-redbank-mcp-server --binary --strategy docker \
        --to="image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/redbank-mcp-server:latest" \
        -n "${NAMESPACE}"
}

echo "  Starting build..."
${KUBECTL} start-build build-redbank-mcp-server --from-dir=. --follow -n "${NAMESPACE}"

# Apply MCP server deployment (patch namespace in the manifest)
echo "  Deploying MCP server..."
sed "s|redbank-demo|${NAMESPACE}|g" mcp-server.yaml | ${KUBECTL} apply -n "${NAMESPACE}" -f -

echo "  Waiting for MCP server to be ready..."
${KUBECTL} wait --for=condition=available --timeout=120s deployment/redbank-mcp-server -n "${NAMESPACE}" || {
    echo "  WARNING: MCP server not ready yet, continuing..."
}

cd "${SCRIPT_DIR}"

# Deploy AgentWorkflow
echo ""
echo "[6/6] Deploying AgentWorkflow..."
# Patch the workflow with the correct namespace
sed "s|namespace: redbank-agentic|namespace: ${NAMESPACE}|g" workflow.yaml | \
    sed "s|redbank-agentic|${NAMESPACE}|g" | \
    ${KUBECTL} apply -f -

echo ""
echo "=============================================="
echo "Deployment initiated!"
echo "=============================================="
echo ""
echo "Monitor progress with:"
echo "  ${KUBECTL} get pods -n ${NAMESPACE} -w"
echo "  ${KUBECTL} get agentworkflow -n ${NAMESPACE} -w"
echo ""
echo "Expected timeline:"
echo "  - PostgreSQL: ~1 min"
echo "  - MCP server build: ~2-3 min"
echo "  - vLLM model download: ~5-10 min"
echo "  - Llama Stack: ~2 min"
echo "  - Vector store ingestion: ~2-3 min"
echo "  - Agents: ~1 min"
echo ""
echo "Test with:"
echo "  ${KUBECTL} port-forward svc/redbank-orchestrator-adapter 8080:8080 -n ${NAMESPACE}"
echo "  curl -X POST http://localhost:8080/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"messages\": [{\"role\": \"user\", \"content\": \"What services does Red Bank offer?\"}]}'"

