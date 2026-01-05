# Base Agent

The Base Agent is the runtime component for all agents deployed by the **rh-agentic-operator**. It provides a unified architecture for building AI agents with retrieval-augmented generation (RAG), external tool access via MCP, and multi-agent orchestration capabilities.

## Features

- **RAG Integration** — Query vector stores using the `file_search` tool
- **MCP Tools** — Connect to external services via Model Context Protocol (GitHub, databases, APIs)
- **A2A Orchestration** — Route requests to specialized subagents and synthesize responses
- **OpenAI Compatibility** — Works with any OpenAI-compatible endpoint (Llama Stack, OpenAI, vLLM, Ollama)
- **MLflow Tracking** — Automatic experiment logging and tracing (optional)
- **Kagenti Integration** — A2A streaming support for the Kagenti UI

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         BASE AGENT                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│   │     RAG      │   │  MCP Tools   │   │ A2A Orchestration│   │
│   │  (Vector     │   │  (External   │   │   (Multi-Agent   │   │
│   │   Stores)    │   │   Services)  │   │    Routing)      │   │
│   └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘   │
│          │                  │                    │              │
│          └──────────────────┼────────────────────┘              │
│                             │                                   │
│                             ▼                                   │
│                 ┌───────────────────────┐                       │
│                 │  OpenAI-Compatible    │                       │
│                 │  LLM Endpoint         │                       │
│                 │  (/v1/responses API)  │                       │
│                 └───────────────────────┘                       │
│                             │                                   │
│                             ▼                                   │
│                 ┌───────────────────────┐                       │
│                 │   MLflow Tracking     │                       │
│                 │   (Optional)          │                       │
│                 └───────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

## Operating Modes

### Simple Mode

When no subagents are configured, the agent operates as a standalone RAG/MCP agent:

```
User Request → [Instruction + Question + Tools] → LLM → Response
```

### Orchestrator Mode

When subagents are configured, the agent operates as an intelligent orchestrator:

```
User Request
     │
     ▼
┌─────────────────────────────────┐
│  1. ROUTING                     │
│  Select relevant subagents      │
│  Craft tailored questions       │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  2. SUBAGENT CALLS              │
│  Execute parallel A2A requests  │
│  Collect Q&A pairs              │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  3. SYNTHESIS                   │
│  Combine agent responses        │
│  Apply RAG/MCP if needed        │
│  Generate final answer          │
└─────────────────────────────────┘
     │
     ▼
Final Response
```

## Instruction System

The Base Agent uses a layered instruction system that ensures reliable orchestration behavior while allowing customization.

### Instruction Types

| Instruction | Scope | Purpose |
|-------------|-------|---------|
| **Routing Instruction** | Internal (fixed) | Ensures proper JSON output for agent selection |
| **Synthesis Instruction** | Internal (fixed) | Ensures proper multi-source answer synthesis |
| **Agent Instruction** | User-defined | Custom behavior, tone, and domain focus |

### Instruction Flow

#### Simple Mode

```
┌────────────────────────────────────────┐
│  SYSTEM INSTRUCTION:                   │
│  {User's spec.instruction}             │
│                                        │
│  USER REQUEST:                         │
│  {User's question}                     │
└────────────────────────────────────────┘
```

#### Orchestrator Mode — Routing Phase

The routing phase uses a fixed instruction to ensure reliable JSON parsing:

```
┌────────────────────────────────────────┐
│  [ROUTING_INSTRUCTION - Fixed]         │
│  - Strict JSON output rules            │
│  - Agent selection criteria            │
│  - Question crafting guidelines        │
│                                        │
│  Available Agents: [...]               │
│  User Question: "..."                  │
└────────────────────────────────────────┘
         │
         ▼
   JSON: {"selected": [...]}
```

#### Orchestrator Mode — Synthesis Phase

The synthesis phase combines the fixed synthesis rules with the user's custom instruction:

```
┌────────────────────────────────────────┐
│  [SYNTHESIS_INSTRUCTION - Fixed]       │
│  - Multi-source synthesis rules        │
│  - Citation requirements               │
│  - Conflict resolution guidelines      │
│                                        │
│  [User's spec.instruction - Custom]    │
│  - Domain-specific guidance            │
│  - Tone and style preferences          │
│                                        │
│  User Question: "..."                  │
│  Agent Responses: [Q&A pairs]          │
│  Tools: [RAG, MCP]                     │
└────────────────────────────────────────┘
         │
         ▼
   Final Synthesized Answer
```

### Built-in Instructions

#### Routing Instruction

Enforces strict JSON output for agent selection:

```
You are an intelligent orchestrator that routes questions to specialized agents.

CRITICAL RULES:
1. Respond with ONLY valid JSON - no markdown, no explanations
2. Analyze the user's question and available agents carefully
3. Select only genuinely relevant agents
4. Craft SPECIFIC questions tailored to each agent's expertise
5. Extract exactly the information needed from each agent
6. Do NOT repeat the user's question verbatim
7. Return empty selection if no agents are relevant

OUTPUT FORMAT:
{
  "selected": [
    {
      "name": "agent-name",
      "question": "specific question for this agent",
      "reason": "why this agent was selected"
    }
  ]
}
```

#### Synthesis Instruction

Ensures comprehensive multi-source answer synthesis:

```
You are an intelligent assistant synthesizing information from multiple sources.

CRITICAL RULES:
1. Answer the user's original question comprehensively
2. Use information from consulted agents when relevant
3. Use RAG/MCP tools for additional information if needed
4. Cite which agent provided which information
5. Be clear, accurate, and well-organized
6. Acknowledge and explain conflicting information
7. Follow any additional instructions provided
```

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_INSTRUCTION` | Custom system prompt for the agent | *(empty)* |
| `OPENAI_BASE_URL` | OpenAI-compatible API endpoint | `http://llama-stack-service:8321` |
| `OPENAI_MODEL` | Model identifier | `vllm-inference-1/qwen3-14b-awq` |
| `OPENAI_API_KEY` | API key for authentication | `not-needed` |
| `VECTOR_STORE_IDS` | Vector store IDs (JSON array or comma-separated) | *(empty)* |
| `MAX_RESULTS` | Maximum results per vector search | `10` |
| `MCP_TOOLS` | MCP server configurations (JSON array) | *(empty)* |
| `A2A_AGENTS_JSON` | Subagent configurations (JSON array) | *(empty)* |
| `A2A_MAX_SELECTED` | Maximum subagents per request | `4` |
| `A2A_CALL_TIMEOUT_S` | Subagent call timeout (seconds) | `60` |
| `A2A_SELECTION_DEFAULT_BATCH` | Default batch for ungrouped agents | `default` |
| `MLFLOW_TRACKING_URI` | MLflow server URI (empty = disabled) | *(empty)* |
| `MLFLOW_EXPERIMENT` | MLflow experiment name | *(agent name)* |
| `MLFLOW_S3_ENDPOINT_URL` | S3 endpoint for MLflow artifacts | *(empty)* |

### MCP Tools Configuration

```json
[
  {
    "server_url": "https://api.githubcopilot.com/mcp/x/repos/readonly",
    "server_label": "GitHub",
    "headers": {
      "Authorization": "Bearer <token>"
    }
  }
]
```

### Subagent Configuration

```json
[
  {
    "name": "docs-agent",
    "url": "http://docs-agent-adapter:8080",
    "skill": "search-docs",
    "batch": "knowledge"
  },
  {
    "name": "code-agent",
    "url": "http://code-agent-adapter:8080",
    "skill": "analyze-code",
    "batch": "technical"
  }
]
```

## API Reference

### POST /v1/chat/completions

OpenAI-compatible chat completion endpoint.

**Request:**
```json
{
  "model": "base-agent",
  "messages": [
    {"role": "user", "content": "Your question here"}
  ],
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Agent response here"
      }
    }
  ]
}
```

**Streaming:** Set `"stream": true` to receive Server-Sent Events (SSE).

### GET /healthz

Health check endpoint.

**Response:**
```json
{
  "ok": true,
  "agent": "agent-name",
  "vector_stores": 2,
  "mcp_tools_count": 1,
  "a2a_agents_count": 3
}
```

### GET /.well-known/agent.json

A2A agent card for discovery.

**Response:**
```json
{
  "name": "agent-name",
  "description": "Agent description",
  "url": "http://agent-endpoint:8080",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "skill-id",
      "name": "Skill Name",
      "description": "Skill description"
    }
  ]
}
```

## MLflow Integration

When `MLFLOW_TRACKING_URI` is configured, the agent automatically logs:

### Artifacts

| Artifact | Description |
|----------|-------------|
| `prompt.txt` | The input prompt sent to the LLM |
| `response.txt` | The generated response |
| `routing.json` | Agent selection decisions (orchestrator mode) |
| `subagent_results.json` | Full Q&A pairs from subagents |
| `subagent_qa.md` | Human-readable Q&A summary |

### Metrics

- Request latency
- Routing latency (orchestrator mode)
- Number of agents called
- Token counts (when available)

## Examples

### RAG Agent

```yaml
apiVersion: agents.redhat.com/v1alpha1
kind: BaseAgent
metadata:
  name: docs-agent
spec:
  instruction: |
    You are a documentation assistant for OpenShift.
    Provide accurate answers based on the documentation.
    Always cite specific documents when possible.
  
  openai:
    baseUrl: http://llama-stack-service:8321
    model: vllm-inference-1/qwen3-14b-awq
  
  rag:
    vectorStoreIds:
      - "vs_openshift_docs"
    maxResults: 10
  
  agentCard:
    description: "OpenShift documentation search agent"
    skills:
      - id: search-docs
        name: Search Documentation
        description: Search OpenShift documentation
```

### MCP Agent

```yaml
apiVersion: agents.redhat.com/v1alpha1
kind: BaseAgent
metadata:
  name: code-agent
spec:
  instruction: |
    You are a code analysis expert.
    Review code quality and suggest improvements.
    Be specific and provide examples.
  
  openai:
    baseUrl: http://llama-stack-service:8321
    model: vllm-inference-1/qwen3-14b-awq
  
  mcpTools:
    - serverUrl: https://api.githubcopilot.com/mcp/x/repos/readonly
      serverLabel: GitHub
      secretRef:
        name: github-token
        key: token
  
  agentCard:
    description: "GitHub code analysis agent"
    skills:
      - id: analyze-code
        name: Analyze Code
        description: Analyze GitHub repositories
```

### Orchestrator Agent

```yaml
apiVersion: agents.redhat.com/v1alpha1
kind: BaseAgent
metadata:
  name: orchestrator
spec:
  instruction: |
    You are a technical support assistant.
    Provide comprehensive answers using all available sources.
    Be clear and well-organized in your responses.
  
  openai:
    baseUrl: http://llama-stack-service:8321
    model: vllm-inference-1/qwen3-14b-awq
  
  subagents:
    - name: docs-agent
      url: http://docs-agent-adapter:8080
      skill: search-docs
      batch: knowledge
    - name: code-agent
      url: http://code-agent-adapter:8080
      skill: analyze-code
      batch: technical
  
  orchestration:
    maxSelected: 4
    callTimeoutSeconds: 60
  
  agentCard:
    description: "Technical support orchestrator"
    skills:
      - id: support
        name: Technical Support
        description: Answer technical questions using multiple sources
```

## Deployment

The Base Agent is deployed automatically by the `rh-agentic-operator` when a `BaseAgent` custom resource is created. The operator handles:

- Building the container image
- Creating the Kubernetes Deployment and Service
- Configuring environment variables from the CR spec
- Setting up MLflow credentials
- Creating Kagenti proxy for UI integration

See the [rh-agentic-operator README](../README.md) for deployment instructions.
