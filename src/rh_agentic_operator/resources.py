"""Resource builders for BaseAgent managed resources."""

import json
from typing import Any, Dict, List, Optional

from . import constants as C


def build_labels(name: str, component: str) -> Dict[str, str]:
    """Build standard labels for a resource."""
    return {
        "app.kubernetes.io/name": name,
        "app.kubernetes.io/instance": name,
        "app.kubernetes.io/component": component,
        "app.kubernetes.io/managed-by": C.OPERATOR_NAME,
    }


def build_owner_reference(owner: Dict[str, Any]) -> Dict[str, Any]:
    """Build owner reference for garbage collection."""
    return {
        "apiVersion": f"{C.API_GROUP}/{C.API_VERSION}",
        "kind": "BaseAgent",
        "name": owner["metadata"]["name"],
        "uid": owner["metadata"]["uid"],
        "controller": True,
        # Note: blockOwnerDeletion requires finalizer permissions
    }


def build_deployment(name: str, namespace: str, spec: Dict[str, Any], owner: Dict[str, Any]) -> Dict[str, Any]:
    """Build Deployment for the base-agent adapter."""
    openai_config = spec.get("openai", {})
    rag = spec.get("rag", {})
    orchestration = spec.get("orchestration", {})
    mlflow = spec.get("mlflow", {})
    image_config = spec.get("image", {})
    resources = spec.get("resources", {})
    replicas = spec.get("replicas", 1)
    
    # Build image reference
    if image_config.get("repository"):
        image_repo = image_config["repository"]
        image_tag = image_config.get("tag", "latest")
        image = f"{image_repo}:{image_tag}"
    else:
        image = C.DEFAULT_BASE_AGENT_IMAGE
    pull_policy = image_config.get("pullPolicy", "Always")
    
    # Environment variables
    openai_base_url = openai_config.get("baseUrl", C.DEFAULT_OPENAI_BASE_URL)
    openai_model = openai_config.get("model", C.DEFAULT_OPENAI_MODEL)
    
    # Agent card metadata
    agent_card = spec.get("agentCard", {})
    agent_description = agent_card.get("description", f"{name} agent")
    
    env = [
        {"name": "HOME", "value": "/tmp"},
        {"name": "AGENT_NAME", "value": name},
        {"name": "AGENT_DESCRIPTION", "value": agent_description},
        {"name": "OPENAI_BASE_URL", "value": openai_base_url},
        {"name": "OPENAI_MODEL", "value": openai_model},
        {"name": "MAX_RESULTS", "value": str(rag.get("maxResults", C.DEFAULT_MAX_RESULTS))},
        {"name": "A2A_MAX_SELECTED", "value": str(orchestration.get("maxSelected", C.DEFAULT_A2A_MAX_SELECTED))},
        {"name": "A2A_CALL_TIMEOUT_S", "value": str(orchestration.get("callTimeoutSeconds", C.DEFAULT_A2A_CALL_TIMEOUT_S))},
        {"name": "A2A_SELECTION_DEFAULT_BATCH", "value": orchestration.get("defaultBatch", C.DEFAULT_A2A_DEFAULT_BATCH)},
    ]
    
    # OpenAI API Key (required for OpenAI/Azure, optional for local endpoints like Llama Stack)
    api_key_ref = openai_config.get("apiKeySecretRef", {})
    if api_key_ref.get("name"):
        env.append({
            "name": "OPENAI_API_KEY",
            "valueFrom": {
                "secretKeyRef": {
                    "name": api_key_ref["name"],
                    "key": api_key_ref.get("key", "api-key"),
                    "optional": True,
                }
            }
        })
    
    # Vector store IDs
    vector_store_ids = rag.get("vectorStoreIds", [])
    if vector_store_ids:
        env.append({"name": "VECTOR_STORE_IDS", "value": json.dumps(vector_store_ids)})
    
    # MCP tools secret reference
    if spec.get("mcpTools"):
        env.append({
            "name": "MCP_TOOLS",
            "valueFrom": {
                "secretKeyRef": {
                    "name": f"{name}-mcp-config",
                    "key": "MCP_TOOLS",
                    "optional": True,
                }
            }
        })
    
    # A2A agents secret reference
    if spec.get("subagents"):
        env.append({
            "name": "A2A_AGENTS_JSON",
            "valueFrom": {
                "secretKeyRef": {
                    "name": f"{name}-a2a-config",
                    "key": "A2A_AGENTS_JSON",
                    "optional": True,
                }
            }
        })
    
    # MLflow configuration
    if mlflow.get("trackingUri"):
        env.append({"name": "MLFLOW_TRACKING_URI", "value": mlflow["trackingUri"]})
        env.append({"name": "MLFLOW_EXPERIMENT", "value": mlflow.get("experiment", name)})
        
        if mlflow.get("s3EndpointUrl"):
            env.append({"name": "MLFLOW_S3_ENDPOINT_URL", "value": mlflow["s3EndpointUrl"]})
        
        creds = mlflow.get("credentialsSecretRef", {})
        if creds.get("name"):
            env.extend([
                {
                    "name": "AWS_ACCESS_KEY_ID",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": creds["name"],
                            "key": creds.get("accessKeyIdKey", "AWS_ACCESS_KEY_ID"),
                            "optional": True,
                        }
                    }
                },
                {
                    "name": "AWS_SECRET_ACCESS_KEY",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": creds["name"],
                            "key": creds.get("secretAccessKeyKey", "AWS_SECRET_ACCESS_KEY"),
                            "optional": True,
                        }
                    }
                },
            ])
    
    # Resource configuration
    resource_config = {
        "requests": {
            "cpu": resources.get("requests", {}).get("cpu", C.DEFAULT_CPU_REQUEST),
            "memory": resources.get("requests", {}).get("memory", C.DEFAULT_MEMORY_REQUEST),
        },
        "limits": {
            "cpu": resources.get("limits", {}).get("cpu", C.DEFAULT_CPU_LIMIT),
            "memory": resources.get("limits", {}).get("memory", C.DEFAULT_MEMORY_LIMIT),
        },
    }
    
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"{name}-adapter",
            "namespace": namespace,
            "labels": build_labels(name, "adapter"),
            "ownerReferences": [build_owner_reference(owner)],
        },
        "spec": {
            "replicas": replicas,
            "selector": {
                "matchLabels": {
                    "app.kubernetes.io/name": name,
                    "app.kubernetes.io/component": "adapter",
                }
            },
            "template": {
                "metadata": {
                    "labels": build_labels(name, "adapter"),
                },
                "spec": {
                    "containers": [{
                        "name": "adapter",
                        "image": image,
                        "imagePullPolicy": pull_policy,
                        "ports": [{"containerPort": C.ADAPTER_PORT, "name": "http"}],
                        "env": env,
                        "readinessProbe": {
                            "httpGet": {"path": "/healthz", "port": C.ADAPTER_PORT},
                            "initialDelaySeconds": 10,
                            "periodSeconds": 10,
                        },
                        "livenessProbe": {
                            "httpGet": {"path": "/healthz", "port": C.ADAPTER_PORT},
                            "initialDelaySeconds": 30,
                            "periodSeconds": 30,
                        },
                        "resources": resource_config,
                    }],
                    "tolerations": [{
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    }],
                },
            },
        },
    }


def build_service(name: str, namespace: str, owner: Dict[str, Any]) -> Dict[str, Any]:
    """Build Service for the adapter."""
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{name}-adapter",
            "namespace": namespace,
            "labels": build_labels(name, "adapter"),
            "ownerReferences": [build_owner_reference(owner)],
        },
        "spec": {
            "selector": {
                "app.kubernetes.io/name": name,
                "app.kubernetes.io/component": "adapter",
            },
            "ports": [{
                "name": "http",
                "port": C.ADAPTER_PORT,
                "targetPort": C.ADAPTER_PORT,
            }],
            "type": "ClusterIP",
        },
    }


def build_mcp_secret(
    name: str,
    namespace: str,
    mcp_tools: List[Dict[str, Any]],
    owner: Dict[str, Any],
    secret_values: Dict[str, str],
) -> Dict[str, Any]:
    """Build Secret containing MCP_TOOLS JSON.
    
    Args:
        name: BaseAgent name
        namespace: Namespace
        mcp_tools: List of MCP tool configs from spec
        owner: Owner reference
        secret_values: Dict mapping secretRef names to their token values
    """
    tools_json = []
    for tool in mcp_tools:
        tool_config = {
            "server_url": tool["serverUrl"],
            "server_label": tool["serverLabel"],
        }
        
        # Add authorization from secret
        secret_ref = tool.get("secretRef", {})
        if secret_ref.get("name"):
            token = secret_values.get(secret_ref["name"])
            if token:
                # Use headers format for Llama Stack
                tool_config["headers"] = {"Authorization": f"Bearer {token}"}
        
        if tool.get("allowedTools"):
            tool_config["allowed_tools"] = tool["allowedTools"]
        
        tools_json.append(tool_config)
    
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": f"{name}-mcp-config",
            "namespace": namespace,
            "labels": build_labels(name, "config"),
            "ownerReferences": [build_owner_reference(owner)],
        },
        "type": "Opaque",
        "stringData": {
            "MCP_TOOLS": json.dumps(tools_json),
        },
    }


def build_a2a_secret(
    name: str,
    namespace: str,
    subagents: List[Dict[str, Any]],
    owner: Dict[str, Any],
) -> Dict[str, Any]:
    """Build Secret containing A2A_AGENTS_JSON."""
    agents_json = []
    for agent in subagents:
        agents_json.append({
            "name": agent["name"],
            "url": agent["url"],
            "skill": agent.get("skill", "answer"),
            "batch": agent.get("batch", "default"),
        })
    
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": f"{name}-a2a-config",
            "namespace": namespace,
            "labels": build_labels(name, "config"),
            "ownerReferences": [build_owner_reference(owner)],
        },
        "type": "Opaque",
        "stringData": {
            "A2A_AGENTS_JSON": json.dumps(agents_json),
        },
    }


def build_model_config(name: str, namespace: str, spec: Dict[str, Any], owner: Dict[str, Any]) -> Dict[str, Any]:
    """Build kagent ModelConfig CR."""
    openai_config = spec.get("openai", {})
    
    return {
        "apiVersion": "kagent.dev/v1alpha2",
        "kind": "ModelConfig",
        "metadata": {
            "name": f"{name}-model",
            "namespace": namespace,
            "labels": build_labels(name, "model"),
            "ownerReferences": [build_owner_reference(owner)],
        },
        "spec": {
            "provider": "OpenAI",
            "model": openai_config.get("model", C.DEFAULT_OPENAI_MODEL),
            "apiKeySecret": f"{name}-key",
            "apiKeySecretKey": "api-key",
            "openAI": {
                # Always point to our adapter which wraps the actual OpenAI-compatible endpoint
                "baseUrl": f"http://{name}-adapter.{namespace}.svc.cluster.local:{C.ADAPTER_PORT}/v1",
            },
        },
    }


def build_api_key_secret(name: str, namespace: str, owner: Dict[str, Any]) -> Dict[str, Any]:
    """Build dummy API key secret for kagent ModelConfig."""
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": f"{name}-key",
            "namespace": namespace,
            "labels": build_labels(name, "config"),
            "ownerReferences": [build_owner_reference(owner)],
        },
        "type": "Opaque",
        "stringData": {
            "api-key": "not-needed",
        },
    }


def build_kagent_agent(name: str, namespace: str, spec: Dict[str, Any], owner: Dict[str, Any]) -> Dict[str, Any]:
    """Build kagent Agent CR."""
    agent_card = spec.get("agentCard", {})
    
    # Build skills for A2A config
    skills = []
    for skill in agent_card.get("skills", []):
        skills.append({
            "id": skill.get("id", "answer"),
            "name": skill.get("name", "Answer"),
            "description": skill.get("description", "Answer questions"),
            "tags": skill.get("tags", []),
        })
    
    # Default skill if none provided
    if not skills:
        skills = [{
            "id": "answer",
            "name": "Answer",
            "description": "Answer questions using RAG, MCP tools, and subagent findings.",
            "tags": ["qa"],
        }]
    
    # Build system message
    system_message = f"""You are {name}, an intelligent assistant.

Capabilities:
- RAG: Access to documentation knowledge base
- MCP: External tools
- A2A: Can consult other specialized agents

Instructions:
- Answer questions clearly and concisely
- Use retrieved context when relevant
- Use tools when they can help answer the question
"""
    
    return {
        "apiVersion": "kagent.dev/v1alpha2",
        "kind": "Agent",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                **build_labels(name, "agent"),
                "kagenti.io/type": "agent",
                "kagenti.io/protocol": "a2a",
                "kagenti.io/framework": "kagent",
            },
            "ownerReferences": [build_owner_reference(owner)],
        },
        "spec": {
            "description": agent_card.get("description", f"BaseAgent: {name}"),
            "type": "Declarative",
            "declarative": {
                "modelConfig": f"{name}-model",
                "stream": False,
                "systemMessage": system_message,
                "a2aConfig": {
                    "skills": skills,
                },
            },
        },
    }

