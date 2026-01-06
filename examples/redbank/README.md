# Redbank Demo (Simplified)

A simplified version of the Redbank demo using the `rh-agentic-operator` to deploy 3 AI agents:

- **redbank-orchestrator**: Routes questions to specialized agents and synthesizes responses
- **redbank-rag-agent**: Answers questions using Redbank documentation (PDFs)
- **redbank-mcp-agent**: Accesses customer data via the Redbank MCP server (PostgreSQL)

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         REDBANK AGENTIC DEMO                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────┐                                                         │
│   │  Kagenti UI    │  ←─ User interacts here                                │
│   │  (or curl)     │                                                         │
│   └───────┬────────┘                                                         │
│           │                                                                  │
│           ▼                                                                  │
│   ┌────────────────┐                                                         │
│   │   redbank-     │  ←─ Routes to redbank-rag-agent and redbank-mcp-agent  │
│   │  orchestrator  │                                                         │
│   └───────┬────────┘                                                         │
│           │                                                                  │
│     ┌─────┴─────┐                                                            │
│     ▼           ▼                                                            │
│ ┌────────────┐ ┌─────────────┐                                               │
│ │ redbank-   │ │ redbank-    │                                               │
│ │ rag-agent  │ │ mcp-agent   │                                               │
│ └─────┬──────┘ └──────┬──────┘                                                    │
│      │            │                                                          │
│      └─────┬──────┘                                                          │
│            ▼                                                                 │
│   ┌────────────────┐      ┌─────────────────┐     ┌─────────────────┐       │
│   │  Llama Stack   │──────│  vLLM (Qwen3)   │     │  redbank-mcp    │       │
│   │  (inference +  │      │  via KServe     │     │  server         │       │
│   │   vector store)│      └─────────────────┘     └────────┬────────┘       │
│   └────────────────┘                                       │                │
│            │                                               ▼                │
│            ▼                                      ┌─────────────────┐       │
│   ┌────────────────┐                              │   PostgreSQL    │       │
│   │ Milvus (inline)│                              │ (customer data) │       │
│   │ Vector Store   │                              └─────────────────┘       │
│   └────────────────┘                                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Files

This example reuses the PostgreSQL and MCP server from `rag/demos/redbank-demo/` to avoid duplication:

```
examples/redbank/
├── README.md          # This file
├── platform.yaml      # AgenticPlatform CR (MLflow + Kagenti)
├── workflow.yaml      # AgentWorkflow CR (model + engine + 3 agents)
├── deploy.sh          # Deployment script (references rag/demos/redbank-demo/)
└── cleanup.sh         # Cleanup script
```

## Prerequisites

### 1. OpenShift Cluster with GPU Support

- OpenShift 4.16+ with admin access
- GPU worker nodes (tested with `g5.2xlarge` / NVIDIA A10G)
- Node Feature Discovery (NFD) Operator installed
- NVIDIA GPU Operator installed with ClusterPolicy

### 2. OpenShift AI / KServe

- OpenShift AI (RHOAI) 3.0+ installed
- KServe enabled (InferenceService CRD available)

### 3. Llama Stack K8s Operator

Enable in RHOAI DSC:

```yaml
spec:
  components:
    llamastackoperator:
      managementState: Managed
```

Verify CRD exists:

```bash
oc get crd llamastackdistributions.llamastack.io
```

### 4. kagent Controller

Install kagent cluster-wide (follow [kagent docs](https://github.com/kagent-dev/kagent)):

```bash
# Verify CRDs exist
oc get crd agents.kagent.dev
oc get crd modelconfigs.kagent.dev
```

## Quick Start

### Step 1: Install the rh-agentic-operator

```bash
cd rh-agentic-operator

# Create operator namespace
oc new-project rh-agentic-system

# Apply CRDs
oc apply -f config/crd.yaml
oc apply -f config/crd-platform.yaml
oc apply -f config/crd-workflow.yaml

# Apply RBAC
cat config/rbac.yaml | sed "s/NAMESPACE_PLACEHOLDER/rh-agentic-system/g" | oc apply -f -

# Create ImageStreams and BuildConfigs
cat << 'EOF' | oc apply -f -
apiVersion: image.openshift.io/v1
kind: ImageStream
metadata:
  name: rh-agentic-operator
  namespace: rh-agentic-system
---
apiVersion: image.openshift.io/v1
kind: ImageStream
metadata:
  name: base-agent
  namespace: rh-agentic-system
---
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: rh-agentic-operator
  namespace: rh-agentic-system
spec:
  output:
    to:
      kind: ImageStreamTag
      name: 'rh-agentic-operator:latest'
  source:
    type: Binary
    binary: {}
  strategy:
    type: Docker
---
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: base-agent
  namespace: rh-agentic-system
spec:
  output:
    to:
      kind: ImageStreamTag
      name: 'base-agent:latest'
  source:
    type: Binary
    binary: {}
  strategy:
    type: Docker
EOF

# Build images
oc start-build rh-agentic-operator --from-dir=. -n rh-agentic-system --follow
oc start-build base-agent --from-dir=base-agent -n rh-agentic-system --follow

# Deploy operator
cat config/deployment.yaml | sed "s/NAMESPACE_PLACEHOLDER/rh-agentic-system/g" | oc apply -f -

# Wait for operator
oc rollout status deployment/rh-agentic-operator -n rh-agentic-system
```

### Step 2: Deploy AgenticPlatform

```bash
oc apply -f examples/redbank/platform.yaml

# Wait for MLflow and Kagenti
oc get pods -n rh-agentic-system -w
oc get routes -n rh-agentic-system
```

### Step 3: Deploy the Redbank Demo

```bash
# Make the script executable
chmod +x examples/redbank/deploy.sh

# Run deployment (default namespace: redbank-agentic)
./examples/redbank/deploy.sh

# Or specify a custom namespace
./examples/redbank/deploy.sh my-redbank-ns
```

The script will:
1. Create the namespace
2. Set up image pull permissions
3. Deploy PostgreSQL with seed data (from `rag/demos/redbank-demo/postgres-db/`)
4. Build and deploy the MCP server (from `rag/demos/redbank-demo/mcp-server/`)
5. Deploy the AgentWorkflow (vLLM, Llama Stack, vector store, 3 agents)

### Step 4: Monitor Deployment

```bash
# Watch pods
oc get pods -n redbank-agentic -w

# Watch workflow status
oc get agentworkflow -n redbank-agentic -w

# Check agent status
oc get baseagents -n redbank-agentic
```

## Usage

### Via curl

```bash
# Port-forward to orchestrator
oc port-forward svc/redbank-orchestrator-adapter 8080:8080 -n redbank-agentic &

# Test RAG (documentation questions)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What services does Red Bank Financial offer?"}]
  }' | jq .choices[0].message.content

# Test MCP (customer data)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Find customer Alice Johnson by email alice.johnson@email.com and tell me her latest balance."}]
  }' | jq .choices[0].message.content

# Test orchestration (combined)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is Red Bank'\''s policy on overdrafts, and does Alice Johnson have any recent transactions that might cause an overdraft?"}]
  }' | jq .choices[0].message.content
```

### Via Kagenti UI

```bash
# Get Kagenti URL
oc get route kagenti-ui -n rh-agentic-system -o jsonpath='{.spec.host}'
```

Open in browser and select `redbank-orchestrator` to chat.

### Example Questions

**RAG (documentation):**
- "What services does Red Bank Financial offer?"
- "How do I open a new account?"
- "What are Red Bank's business hours?"

**MCP (customer data):**
- "Find customer by email alice.johnson@email.com"
- "Show me Bob Smith's recent transactions"
- "What is the balance for customer ID 3?"

**Orchestration (combined):**
- "Compare Red Bank's FAQ policies with Alice Johnson's recent transactions"
- "Does Red Bank offer services that would help David Brown's business account?"

## Observability

### MLflow

View agent traces and logs:

```bash
oc get route mlflow -n rh-agentic-system -o jsonpath='{.spec.host}'
```

## Cleanup

```bash
chmod +x examples/redbank/cleanup.sh
./examples/redbank/cleanup.sh

# Or for a specific namespace
./examples/redbank/cleanup.sh my-redbank-ns
```

## Troubleshooting

### Model not loading

```bash
oc logs -l serving.kserve.io/inferenceservice=qwen3-14b-awq -n redbank-agentic --tail=100
```

### Vector store job failing

```bash
oc logs job/redbank-workflow-vectorstore-setup -n redbank-agentic
```

### MCP agent not finding data

```bash
# Check MCP server
oc logs deployment/redbank-mcp-server -n redbank-agentic

# Check PostgreSQL
oc exec -it deployment/postgresql -n redbank-agentic -- psql -U user -d db -c "SELECT * FROM customers;"
```

### Agent not responding

```bash
oc logs deployment/redbank-orchestrator-adapter -n redbank-agentic
```
