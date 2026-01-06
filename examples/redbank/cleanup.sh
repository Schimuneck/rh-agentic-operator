#!/bin/bash
# Redbank Demo Cleanup Script
#
# Usage: ./cleanup.sh [NAMESPACE]
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
echo "Redbank Agentic Demo Cleanup"
echo "=============================================="
echo "Namespace: ${NAMESPACE}"
echo "=============================================="

echo ""
echo "[1/4] Deleting AgentWorkflow..."
${KUBECTL} delete agentworkflow redbank-workflow -n "${NAMESPACE}" --ignore-not-found

echo ""
echo "[2/4] Deleting MCP server..."
${KUBECTL} delete -f "${REDBANK_DEMO_DIR}/mcp-server/mcp-server.yaml" -n "${NAMESPACE}" --ignore-not-found
${KUBECTL} delete buildconfig build-redbank-mcp-server -n "${NAMESPACE}" --ignore-not-found

echo ""
echo "[3/4] Deleting PostgreSQL..."
${KUBECTL} delete -k "${REDBANK_DEMO_DIR}/postgres-db/" -n "${NAMESPACE}" --ignore-not-found

echo ""
echo "[4/4] Deleting namespace..."
read -p "Delete namespace ${NAMESPACE}? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ${KUBECTL} delete namespace "${NAMESPACE}"
    echo "Namespace deleted"
else
    echo "Namespace kept"
fi

echo ""
echo "Cleanup complete!"

