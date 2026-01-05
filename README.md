# rh-agentic-operator

A Kubernetes operator for deploying and managing AI agents at scale. Define agents as YAML, deploy them anywhere in your cluster, and let the operator handle MLflow tracking, Kagenti UI discovery, and A2A orchestration automatically.

## Goal

**Simplify AI agent deployment on Kubernetes/OpenShift** by providing:

- **One YAML per agent** — Deploy RAG, MCP, or orchestrator agents with a single CR
- **Custom instructions** — Define agent behavior with system prompts
- **Automatic platform setup** — MLflow, Kagenti UI deployed and configured automatically
- **Cross-namespace management** — Cluster-wide operator reconciles agents in any namespace
- **Kagent integration** — Automatic A2A protocol support via kagent framework
- **Full observability** — MLflow tracks prompts, responses, routing decisions, and subagent Q&A

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPERATOR NAMESPACE (rh-agentic-system)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐  ┌────────────────┐  ┌────────────────────┐   │
│  │  rh-agentic-operator    │  │  MLflow Stack  │  │  Kagenti UI        │   │
│  │  (cluster-wide)         │  │  - mlflow      │  │  - agent discovery │   │
│  │                         │  │  - minio       │  │  - cluster-wide    │   │
│  │  Watches: BaseAgent     │  │  - postgres    │  │    agent listing   │   │
│  │  across all namespaces  │  │                │  │                    │   │
│  └─────────────────────────┘  └────────────────┘  └────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                   Reconciles agents in any namespace
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AGENT NAMESPACE (any namespace)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐   │
│  │  rag-agent        │  │  mcp-agent        │  │  orchestrator-agent   │   │
│  │  ├─ adapter       │  │  ├─ adapter       │  │  ├─ adapter           │   │
│  │  ├─ kagent agent  │  │  ├─ kagent agent  │  │  ├─ kagent agent      │   │
│  │  └─ kagenti proxy │  │  └─ kagenti proxy │  │  └─ kagenti proxy     │   │
│  │                   │  │                   │  │                       │   │
│  │  Capabilities:    │  │  Capabilities:    │  │  Capabilities:        │   │
│  │  - Vector search  │  │  - GitHub MCP     │  │  - Call subagents     │   │
│  │  - RAG retrieval  │  │  - Code search    │  │  - Smart routing      │   │
│  └───────────────────┘  └───────────────────┘  └───────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ OpenAI-compatible API
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXTERNAL (bring your own)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐   │
│  │  LLM Endpoint     │  │  Vector Stores    │  │  MCP Servers          │   │
│  │  - OpenAI         │  │  - Llama Stack    │  │  - GitHub Copilot     │   │
│  │  - Llama Stack    │  │  - OpenAI         │  │  - Slack              │   │
│  │  - vLLM           │  │                   │  │  - Custom             │   │
│  │  - Ollama         │  │                   │  │                       │   │
│  └───────────────────┘  └───────────────────┘  └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Requirements

### Required

1. **OpenShift/Kubernetes cluster** with admin access
2. **kagent** — Agent framework controller (cluster-wide)
   - Provides: `agents.kagent.dev`, `modelconfigs.kagent.dev` CRDs
   - Install: https://github.com/kagent-dev/kagent
   - Typically in `kagent` namespace

3. **OpenAI-compatible LLM endpoint** — One of:
   - OpenAI API (`https://api.openai.com/v1`)
   - Llama Stack (`http://llama-stack-service:8321`)
   - vLLM (`http://vllm-service:8000/v1`)
   - Ollama (`http://ollama:11434/v1`)
   - Azure OpenAI

### Optional

- **Kagenti CRDs** — For UI discovery (`agents.agent.kagenti.dev`)
- **Vector stores** — For RAG (in Llama Stack or OpenAI)
- **MCP servers** — For tool integrations (GitHub, Slack, etc.)

## Installation

### Step 1: Create Operator Namespace

```bash
# Create namespace for operator and platform services
oc new-project rh-agentic-system

# Or with kubectl:
kubectl create namespace rh-agentic-system
```

### Step 2: Deploy CRDs and RBAC

```bash
cd rh-agentic-operator

# Apply CRDs
oc apply -f config/crd.yaml
oc apply -f config/crd-platform.yaml

# Apply RBAC (replace namespace)
cat config/rbac.yaml | sed "s/NAMESPACE_PLACEHOLDER/rh-agentic-system/g" | oc apply -f -
```

### Step 3: Build and Deploy Operator

```bash
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
```

### Step 4: Deploy Platform (MLflow + Kagenti)

```yaml
# platform.yaml
apiVersion: agents.redhat.com/v1alpha1
kind: AgenticPlatform
metadata:
  name: agentic-platform
  namespace: rh-agentic-system
spec:
  # Default LLM for all agents (can be overridden per agent)
  openaiDefaults:
    baseUrl: http://llama-stack-service.agentic-test.svc.cluster.local:8321
    model: vllm-shared/qwen3-14b-awq
  
  # Platform components to deploy
  components:
    mlflow:
      enabled: true
      storage:
        size: 10Gi
    kagenti:
      enabled: true
    kagent:
      enabled: true  # Will skip if already installed cluster-wide
  
  # Optional: Override default images
  images:
    baseAgent: image-registry.openshift-image-registry.svc:5000/rh-agentic-system/base-agent:latest
```

```bash
oc apply -f platform.yaml
```

## Deploying Agents

### Create Agent Namespace

```bash
# Create namespace for your agents
oc new-project agentic-workloads

# Allow image pulls from operator namespace
oc policy add-role-to-user system:image-puller \
  system:serviceaccount:agentic-workloads:default \
  -n rh-agentic-system

# Label for Kagenti discovery
oc label namespace agentic-workloads kagenti-enabled=true
```

### RAG Agent

Answers questions using a vector store for retrieval-augmented generation.

```yaml
# rag-agent.yaml
apiVersion: agents.redhat.com/v1alpha1
kind: BaseAgent
metadata:
  name: rag-agent
  namespace: agentic-workloads
spec:
  # Agent instruction (system prompt)
  instruction: |
    You are a documentation assistant.
    Search the knowledge base to answer questions accurately.
    Always cite specific documents when possible.
    If information is not found, clearly state that.
  
  # RAG configuration
  rag:
    vectorStoreIds:
      - "docs-vectorstore"      # Vector store ID in Llama Stack
    maxResults: 10              # Max documents to retrieve
  
  # Agent metadata for A2A discovery
  agentCard:
    description: "Documentation agent - answers questions using the knowledge base"
    skills:
      - id: answer
        name: Answer Questions
        description: "Answer questions using RAG retrieval from documentation"
        tags: ["qa", "rag", "documentation"]
```

```bash
oc apply -f rag-agent.yaml
```

### MCP Agent (GitHub)

Accesses external tools via Model Context Protocol (MCP).

```yaml
# First, create the secret with your GitHub token
apiVersion: v1
kind: Secret
metadata:
  name: github-token
  namespace: agentic-workloads
type: Opaque
stringData:
  token: "ghp_your_github_personal_access_token"
---
# mcp-agent.yaml
apiVersion: agents.redhat.com/v1alpha1
kind: BaseAgent
metadata:
  name: mcp-agent
  namespace: agentic-workloads
spec:
  # Agent instruction (system prompt)
  instruction: |
    You are a code analysis expert with access to GitHub.
    Search repositories, review code, and provide insights.
    Be specific and include code examples when helpful.
  
  # MCP tools configuration
  mcpTools:
    - serverUrl: "https://api.githubcopilot.com/mcp/x/repos/readonly"
      serverLabel: "GitHub"
      secretRef:
        name: github-token
        key: token
  
  agentCard:
    description: "Code agent - searches and analyzes GitHub repositories"
    skills:
      - id: github
        name: GitHub Operations
        description: "Search repositories, list commits, get file contents"
        tags: ["github", "code", "repository"]
```

```bash
oc apply -f mcp-agent.yaml
```

### Orchestrator Agent

Coordinates multiple specialized agents using A2A protocol.

```yaml
# orchestrator-agent.yaml
apiVersion: agents.redhat.com/v1alpha1
kind: BaseAgent
metadata:
  name: orchestrator-agent
  namespace: agentic-workloads
spec:
  # Agent instruction (custom behavior added to synthesis)
  instruction: |
    You are a technical support assistant.
    Provide comprehensive answers using all available sources.
    Be clear, accurate, and well-organized.
  
  # Subagents to coordinate
  subagents:
    - name: rag-agent
      url: http://rag-agent.agentic-workloads.svc.cluster.local:8080
      skill: answer
      batch: documentation  # Group for LLM selection
    - name: mcp-agent
      url: http://mcp-agent.agentic-workloads.svc.cluster.local:8080
      skill: github
      batch: code
  
  # Orchestration settings
  orchestration:
    maxSelected: 2            # Max agents to call per request
    callTimeoutSeconds: 60    # Timeout for subagent calls
  
  agentCard:
    description: "Orchestrator - coordinates documentation and code agents"
    skills:
      - id: orchestrate
        name: Orchestrate
        description: "Route questions to specialized agents and synthesize responses"
        tags: ["orchestrator", "multi-agent"]
```

```bash
oc apply -f orchestrator-agent.yaml
```

### Using OpenAI Directly

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: openai-api-key
  namespace: agentic-workloads
type: Opaque
stringData:
  api-key: "sk-your-openai-api-key"
---
apiVersion: agents.redhat.com/v1alpha1
kind: BaseAgent
metadata:
  name: openai-agent
  namespace: agentic-workloads
spec:
  instruction: |
    You are a helpful AI assistant.
    Provide accurate and concise answers.
  
  openai:
    baseUrl: https://api.openai.com/v1
    model: gpt-4o
    apiKeySecretRef:
      name: openai-api-key
      key: api-key
  
  agentCard:
    description: "Agent powered by OpenAI GPT-4o"
    skills:
      - id: chat
        name: Chat
        description: "General conversation"
```

## MLflow Tracking

Every agent automatically logs to MLflow when `MLFLOW_TRACKING_URI` is configured (automatic when using `AgenticPlatform`).

### What Gets Tracked

| Artifact | Description |
|----------|-------------|
| `prompt.txt` | The full prompt sent to the LLM |
| `response.txt` | The generated response |
| `routing.json` | Agent routing decisions (orchestrator only) |
| `subagent_results.json` | Full Q&A pairs from subagents (orchestrator only) |
| `subagent_qa.md` | Human-readable Q&A summary (orchestrator only) |

### Orchestrator Routing Tracking

For orchestrator agents, MLflow captures the complete routing decision:

```json
{
  "routing_ms": 1234,
  "candidates": 5,
  "batches": ["documentation", "code"],
  "selected": [
    {
      "name": "rag-agent",
      "question": "What are the deployment requirements for OpenShift?",
      "reason": "User asked about deployment, RAG agent has documentation"
    },
    {
      "name": "mcp-agent", 
      "question": "List recent commits in the deployment repository",
      "reason": "User mentioned code changes, MCP agent can access GitHub"
    }
  ]
}
```

### Subagent Q&A Tracking

The orchestrator also logs all Q&A exchanges with subagents:

```json
{
  "qa_pairs": [
    {
      "name": "rag-agent",
      "batch": "documentation",
      "question": "What are the deployment requirements for OpenShift?",
      "reason": "User asked about deployment",
      "answer": "OpenShift requires...",
      "elapsed_ms": 2500
    },
    {
      "name": "mcp-agent",
      "batch": "code",
      "question": "List recent commits in the deployment repository",
      "reason": "User mentioned code changes",
      "answer": "Recent commits include...",
      "elapsed_ms": 3200
    }
  ]
}
```

### Viewing in MLflow UI

1. Open MLflow UI: `https://mlflow-<namespace>.<cluster-domain>`
2. Select the experiment (agent name)
3. Click on a run to see:
   - **Parameters**: Agent configuration
   - **Artifacts**: Prompt, response, routing, Q&A logs
   - **Metrics**: Latency, token counts

## What Gets Created

For each `BaseAgent`, the operator creates:

| Resource | Name | Description |
|----------|------|-------------|
| Deployment | `<name>-adapter` | The agent adapter (FastAPI server) |
| Service | `<name>-adapter` | ClusterIP service for the adapter |
| Secret | `<name>-key` | API key for kagent |
| Secret | `<name>-mcp-config` | MCP tools configuration (if mcpTools specified) |
| Secret | `<name>-a2a-config` | Subagent configuration (if subagents specified) |
| Secret | `mlflow-credentials` | MLflow/MinIO credentials (replicated from platform) |
| ModelConfig | `<name>-model` | kagent model configuration |
| Agent (kagent) | `<name>` | kagent Agent CR with A2A support |
| ConfigMap | `<name>-proxy-config` | Nginx config for Kagenti proxy |
| Agent (kagenti) | `<name>-ui` | Kagenti Agent CR for UI discovery |

## Platform URLs

After deploying `AgenticPlatform`, access these UIs:

| Service | URL Pattern |
|---------|-------------|
| **Kagenti UI** | `https://kagenti-ui-<operator-ns>.<cluster-domain>` |
| **MLflow** | `https://mlflow-<operator-ns>.<cluster-domain>` |
| **Kagent UI** | `https://kagent-ui-kagent.<cluster-domain>` (cluster-wide) |

Example:
- Kagenti: `https://kagenti-ui-rh-agentic-system.apps.rosa.mschimun.072j.p3.openshiftapps.com`
- MLflow: `https://mlflow-rh-agentic-system.apps.rosa.mschimun.072j.p3.openshiftapps.com`

## Configuration Reference

### BaseAgent Spec

| Field | Description | Default |
|-------|-------------|---------|
| `instruction` | System prompt defining agent behavior | - |
| `openai.baseUrl` | OpenAI-compatible API URL | From platform |
| `openai.model` | Model ID | From platform |
| `openai.apiKeySecretRef` | Secret with API key | - |
| `rag.vectorStoreIds` | Vector store IDs | `[]` |
| `rag.maxResults` | Max documents to retrieve | `10` |
| `mcpTools[].serverUrl` | MCP server URL | - |
| `mcpTools[].serverLabel` | Display name | - |
| `mcpTools[].secretRef` | Secret with auth token | - |
| `subagents[].name` | Subagent name | - |
| `subagents[].url` | Subagent A2A URL | - |
| `subagents[].skill` | Skill to invoke | `answer` |
| `subagents[].batch` | Batch group for routing | `default` |
| `orchestration.maxSelected` | Max agents per request | `4` |
| `orchestration.callTimeoutSeconds` | Subagent call timeout | `60` |
| `mlflow.trackingUri` | MLflow URL | From platform |
| `mlflow.experiment` | Experiment name | Agent name |
| `agentCard.description` | Agent description | - |
| `agentCard.skills` | A2A skills list | - |
| `kagenti.enabled` | Enable Kagenti proxy | `true` |
| `replicas` | Number of adapter replicas | `1` |

### AgenticPlatform Spec

| Field | Description | Default |
|-------|-------------|---------|
| `openaiDefaults.baseUrl` | Default LLM URL for agents | - |
| `openaiDefaults.model` | Default model for agents | - |
| `components.mlflow.enabled` | Deploy MLflow stack | `true` |
| `components.mlflow.storage.size` | MinIO storage size | `10Gi` |
| `components.kagenti.enabled` | Deploy Kagenti UI | `true` |
| `components.kagent.enabled` | Deploy kagent (if not present) | `false` |
| `images.baseAgent` | Base agent image | - |
| `images.mlflow` | MLflow image | - |
| `images.kagentiUI` | Kagenti UI image | - |

## Troubleshooting

### Check Operator Logs

```bash
oc logs deployment/rh-agentic-operator -n rh-agentic-system -f
```

### Check Agent Status

```bash
# BaseAgent status
oc get baseagents -n agentic-workloads

# Kagent agents
oc get agents.kagent.dev -n agentic-workloads

# Kagenti agents
oc get agents.agent.kagenti.dev -n agentic-workloads

# Pods
oc get pods -n agentic-workloads
```

### Common Issues

1. **ImagePullBackOff** — Agent namespace needs image-puller role:
   ```bash
   oc policy add-role-to-user system:image-puller \
     system:serviceaccount:<agent-ns>:default -n rh-agentic-system
   ```

2. **MLflow bucket error** — Create the bucket in MinIO:
   ```bash
   oc exec -n rh-agentic-system deploy/mlflow-minio -- mkdir -p /data/mlflow
   ```

3. **Kagenti not listing agents** — Ensure namespace is labeled:
   ```bash
   oc label namespace <agent-ns> kagenti-enabled=true
   ```

4. **Model not found** — Check Llama Stack models:
   ```bash
   curl http://llama-stack-service:8321/v1/models | jq '.data[].identifier'
   ```

5. **Context length exceeded** — MCP tools returning too much data:
   - Increase model context length in vLLM (`--max-model-len=65536`)
   - Or use more specific queries to reduce response size

## License

Apache 2.0
