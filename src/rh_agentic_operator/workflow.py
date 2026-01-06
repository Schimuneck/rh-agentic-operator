"""Resource builders for AgentWorkflow managed resources.

This module handles the reconciliation of:
- KServe ServingRuntime and InferenceService (vLLM model serving)
- LlamaStackDistribution and ConfigMap (AI engine)
- VectorStore ingestion Job
- OTEL Collector for observability
- MLflow OTEL Bridge for native traces
"""

import json
import os
from typing import Any, Dict, List, Optional

from . import constants as C

# =============================================================================
# Constants for AgentWorkflow
# =============================================================================

LLAMA_STACK_PORT = 8321
VLLM_PORT = 8080
OTEL_GRPC_PORT = 4317
OTEL_HTTP_PORT = 4318

# Default images
OTEL_COLLECTOR_IMAGE = os.getenv(
    "OTEL_COLLECTOR_IMAGE",
    "otel/opentelemetry-collector-contrib:latest"
)
VECTORSTORE_SETUP_IMAGE = os.getenv(
    "VECTORSTORE_SETUP_IMAGE",
    "python:3.11-slim"
)
MLFLOW_OTEL_BRIDGE_IMAGE = os.getenv(
    "MLFLOW_OTEL_BRIDGE_IMAGE",
    "python:3.11-slim"
)

# =============================================================================
# Label Helpers
# =============================================================================

def build_workflow_labels(workflow_name: str, component: str) -> Dict[str, str]:
    """Build standard labels for workflow-managed resources."""
    return {
        "app.kubernetes.io/name": workflow_name,
        "app.kubernetes.io/instance": workflow_name,
        "app.kubernetes.io/component": component,
        "app.kubernetes.io/managed-by": C.OPERATOR_NAME,
        "agents.redhat.com/workflow": workflow_name,
    }


def build_owner_reference(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Build owner reference for garbage collection."""
    return {
        "apiVersion": f"{C.API_GROUP}/{C.API_VERSION}",
        "kind": "AgentWorkflow",
        "name": workflow["metadata"]["name"],
        "uid": workflow["metadata"]["uid"],
        "controller": True,
    }


# =============================================================================
# KServe Resource Builders (vLLM Model Serving)
# =============================================================================

def build_serving_runtime(
    workflow_name: str,
    namespace: str,
    model_spec: Dict[str, Any],
    workflow: Dict[str, Any],
) -> Dict[str, Any]:
    """Build KServe ServingRuntime for vLLM."""
    model_name = model_spec["name"]
    vllm_config = model_spec.get("vllm", {})
    
    # Determine image
    if vllm_config.get("buildInCluster", True):
        # Use in-cluster built image
        image = f"image-registry.openshift-image-registry.svc:5000/{namespace}/vllm-openai-otel:latest"
    else:
        image = vllm_config.get("image", "vllm/vllm-openai:latest")
    
    return {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": model_name,
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "serving-runtime"),
            "ownerReferences": [build_owner_reference(workflow)],
            "annotations": {
                "opendatahub.io/apiProtocol": "REST",
                "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
                "opendatahub.io/template-name": "vllm-cuda-runtime-template",
            },
        },
        "spec": {
            "annotations": {
                "opendatahub.io/kserve-runtime": "vllm",
                "prometheus.io/path": "/metrics",
                "prometheus.io/port": str(VLLM_PORT),
            },
            "containers": [{
                "name": "kserve-container",
                "image": image,
                "command": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
                "args": [
                    f"--port={VLLM_PORT}",
                    "--model=/mnt/models",
                    "--served-model-name={{.Name}}",
                ],
                "env": [
                    {"name": "HF_HOME", "value": "/tmp/hf_home"},
                    {"name": "HOME", "value": "/tmp"},
                    {"name": "VLLM_USAGE_SOURCE", "value": "production-docker-image"},
                ],
                "ports": [{"containerPort": VLLM_PORT, "protocol": "TCP"}],
            }],
            "multiModel": False,
            "supportedModelFormats": [{
                "autoSelect": True,
                "name": "vLLM",
            }],
        },
    }


def build_inference_service(
    workflow_name: str,
    namespace: str,
    model_spec: Dict[str, Any],
    workflow: Dict[str, Any],
) -> Dict[str, Any]:
    """Build KServe InferenceService for vLLM model."""
    model_name = model_spec["name"]
    storage_uri = model_spec["storageUri"]
    vllm_config = model_spec.get("vllm", {})
    resources = vllm_config.get("resources", {})
    
    # Default resources with GPU
    default_limits = {
        "cpu": "4",
        "memory": "8Gi",
        "nvidia.com/gpu": "1",
    }
    default_requests = {
        "cpu": "2",
        "memory": "4Gi",
        "nvidia.com/gpu": "1",
    }
    
    limits = {**default_limits, **resources.get("limits", {})}
    requests = {**default_requests, **resources.get("requests", {})}
    
    # Build args from vLLM config
    args = vllm_config.get("args", [
        "--dtype=auto",
        "--max-model-len=65536",
        "--enable-auto-tool-choice",
        "--tool-call-parser=hermes",
        "--gpu-memory-utilization=0.90",
    ])
    
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "inference-service"),
            "ownerReferences": [build_owner_reference(workflow)],
            "annotations": {
                "opendatahub.io/genai-use-case": "chat",
                "opendatahub.io/model-type": "generative",
                "openshift.io/display-name": model_name,
                "security.opendatahub.io/enable-auth": "false",
                "serving.kserve.io/deploymentMode": "RawDeployment",
            },
        },
        "spec": {
            "predictor": {
                "automountServiceAccountToken": False,
                "maxReplicas": 1,
                "minReplicas": 1,
                "model": {
                    "args": args,
                    "modelFormat": {"name": "vLLM"},
                    "name": "",
                    "resources": {
                        "limits": limits,
                        "requests": requests,
                    },
                    "runtime": model_name,
                    "storageUri": storage_uri,
                },
            },
        },
    }


# =============================================================================
# Llama Stack Resource Builders
# =============================================================================

def build_llamastack_config(
    workflow_name: str,
    namespace: str,
    workflow_spec: Dict[str, Any],
    workflow: Dict[str, Any],
    platform_namespace: str = "rh-agentic-system",
) -> Dict[str, Any]:
    """Build ConfigMap for Llama Stack run.yaml configuration."""
    model_spec = workflow_spec.get("model", {})
    engine_spec = workflow_spec.get("engine", {})
    observability = workflow_spec.get("observability", {})
    
    model_name = model_spec.get("name", "qwen3-14b-awq")
    model_id = model_spec.get("modelId", model_name)
    
    # Determine model URL: use external URL if specified, otherwise local KServe deployment
    external_url = model_spec.get("externalUrl")
    if external_url:
        vllm_url = external_url
    else:
        vllm_url = f"http://{model_name}-predictor.{namespace}.svc.cluster.local:{VLLM_PORT}/v1"
    
    # Build the run.yaml configuration
    run_config = {
        "version": "2",
        "image_name": "rh",
        "apis": [
            "agents",
            "datasetio",
            "files",
            "inference",
            "safety",
            "scoring",
            "tool_runtime",
            "vector_io",
        ],
        "providers": {
            "inference": [
                {
                    "provider_id": "sentence-transformers",
                    "provider_type": "inline::sentence-transformers",
                    "config": {},
                },
                {
                    "provider_id": "vllm-inference-1",
                    "provider_type": "remote::vllm",
                    "config": {
                        "api_token": "${env.VLLM_API_TOKEN_1:=fake}",
                        "max_tokens": "${env.VLLM_MAX_TOKENS:=4096}",
                        "tls_verify": "${env.VLLM_TLS_VERIFY:=false}",
                        "url": vllm_url,
                    },
                },
            ],
            "vector_io": [{
                "provider_id": "milvus",
                "provider_type": "inline::milvus",
                "config": {
                    "db_path": "/opt/app-root/src/.llama/distributions/rh/milvus.db",
                    "kvstore": {
                        "db_path": "/opt/app-root/src/.llama/distributions/rh/milvus_registry.db",
                        "namespace": None,
                        "type": "sqlite",
                    },
                },
            }],
            "agents": [{
                "provider_id": "meta-reference",
                "provider_type": "inline::meta-reference",
                "config": {
                    "persistence_store": {
                        "db_path": "/opt/app-root/src/.llama/distributions/rh/agents_store.db",
                        "namespace": None,
                        "type": "sqlite",
                    },
                    "responses_store": {
                        "db_path": "/opt/app-root/src/.llama/distributions/rh/responses_store.db",
                        "type": "sqlite",
                    },
                },
            }],
            "eval": [],
            "files": [{
                "provider_id": "meta-reference-files",
                "provider_type": "inline::localfs",
                "config": {
                    "metadata_store": {
                        "db_path": "/opt/app-root/src/.llama/distributions/rh/files_metadata.db",
                        "type": "sqlite",
                    },
                    "storage_dir": "/opt/app-root/src/.llama/distributions/rh/files",
                },
            }],
            "datasetio": [{
                "provider_id": "huggingface",
                "provider_type": "remote::huggingface",
                "config": {
                    "kvstore": {
                        "db_path": "/opt/app-root/src/.llama/distributions/rh/huggingface_datasetio.db",
                        "namespace": None,
                        "type": "sqlite",
                    },
                },
            }],
            "scoring": [
                {"provider_id": "basic", "provider_type": "inline::basic", "config": {}},
                {"provider_id": "llm-as-judge", "provider_type": "inline::llm-as-judge", "config": {}},
            ],
            "tool_runtime": [
                {"provider_id": "rag-runtime", "provider_type": "inline::rag-runtime", "config": {}},
                {"provider_id": "model-context-protocol", "provider_type": "remote::model-context-protocol", "config": {}},
            ],
        },
        "metadata_store": {
            "type": "sqlite",
            "db_path": "/opt/app-root/src/.llama/distributions/rh/inference_store.db",
        },
        "models": [
            {
                "provider_id": "sentence-transformers",
                "model_id": "granite-embedding-125m",
                "provider_model_id": "ibm-granite/granite-embedding-125m-english",
                "model_type": "embedding",
                "metadata": {"embedding_dimension": 768},
            },
            {
                "provider_id": "vllm-inference-1",
                "model_id": model_id,
                "model_type": "llm",
                "metadata": {
                    "description": "",
                    "display_name": model_id,
                },
            },
        ],
        "shields": [],
        "vector_dbs": [],
        "datasets": [],
        "scoring_fns": [],
        "benchmarks": [],
        "tool_groups": [
            {"toolgroup_id": "builtin::rag", "provider_id": "rag-runtime"},
        ],
        "server": {"port": LLAMA_STACK_PORT},
    }
    
    import yaml
    run_yaml = yaml.dump(run_config, default_flow_style=False, sort_keys=False)
    
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "llama-stack-config",
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "llama-stack-config"),
            "ownerReferences": [build_owner_reference(workflow)],
        },
        "data": {
            "run.yaml": run_yaml,
        },
    }


def build_llamastack_distribution(
    workflow_name: str,
    namespace: str,
    workflow_spec: Dict[str, Any],
    workflow: Dict[str, Any],
    platform_namespace: str = "rh-agentic-system",
) -> Dict[str, Any]:
    """Build LlamaStackDistribution CR."""
    engine_spec = workflow_spec.get("engine", {})
    observability = workflow_spec.get("observability", {})
    
    engine_name = engine_spec.get("name", "llama-stack")
    distribution_name = engine_spec.get("distributionName", "rh-dev")
    replicas = engine_spec.get("replicas", 1)
    resources = engine_spec.get("resources", {})
    
    # Default resources (increased for OTEL bootstrap)
    default_limits = {"cpu": "4", "memory": "16Gi"}
    default_requests = {"cpu": "500m", "memory": "2Gi"}
    
    limits = {**default_limits, **resources.get("limits", {})}
    requests = {**default_requests, **resources.get("requests", {})}
    
    # Environment variables
    env = [
        {"name": "VLLM_TLS_VERIFY", "value": "false"},
        {"name": "MILVUS_DB_PATH", "value": "~/.llama/milvus.db"},
        {"name": "FMS_ORCHESTRATOR_URL", "value": "http://localhost"},
        {"name": "VLLM_MAX_TOKENS", "value": "4096"},
        {"name": "VLLM_API_TOKEN_1", "value": "fake"},
        {"name": "LLAMA_STACK_CONFIG_DIR", "value": "/opt/app-root/src/.llama/distributions/rh/"},
        # Ensure pip-installed binaries are in PATH
        {"name": "PATH", "value": "/opt/app-root/lib64/python3.12/site-packages/.local/bin:/opt/app-root/bin:/usr/local/bin:/usr/bin:/bin"},
    ]
    
    # Determine startup command - with or without OTEL instrumentation
    otel_enabled = observability.get("otelEnabled", True)
    
    if otel_enabled:
        # Add OTEL configuration
        env.extend([
            {"name": "OTEL_EXPORTER_OTLP_ENDPOINT", "value": f"http://otel-collector.{namespace}.svc.cluster.local:{OTEL_HTTP_PORT}"},
            {"name": "OTEL_EXPORTER_OTLP_PROTOCOL", "value": "http/protobuf"},
            {"name": "OTEL_SERVICE_NAME", "value": "llama-stack"},
            # Disable problematic instrumentations that can cause issues
            {"name": "OTEL_PYTHON_DISABLED_INSTRUMENTATIONS", "value": "sqlite3"},
        ])
        
        # Command with OTEL instrumentation (with robust fallback):
        # Try OTEL instrumentation, but if it fails, run without it
        # This prevents startup failures due to OTEL issues
        startup_command = (
            "if pip install --quiet opentelemetry-distro opentelemetry-exporter-otlp 2>/dev/null && "
            "timeout 120 opentelemetry-bootstrap -a install 2>/dev/null; then "
            "echo 'OTEL enabled'; "
            "opentelemetry-instrument llama stack run /etc/llama-stack/run.yaml; "
            "else "
            "echo 'OTEL bootstrap failed or timed out, running without instrumentation'; "
            "llama stack run /etc/llama-stack/run.yaml; "
            "fi"
        )
    else:
        # Without OTEL - direct startup
        startup_command = "llama stack run /etc/llama-stack/run.yaml"
    
    return {
        "apiVersion": "llamastack.io/v1alpha1",
        "kind": "LlamaStackDistribution",
        "metadata": {
            "name": engine_name,
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "llamastack"),
            "ownerReferences": [build_owner_reference(workflow)],
            "annotations": {
                "openshift.io/display-name": engine_name,
            },
        },
        "spec": {
            "replicas": replicas,
            "server": {
                "containerSpec": {
                    "command": ["/bin/sh", "-c", startup_command],
                    "env": env,
                    "name": "llama-stack",
                    "port": LLAMA_STACK_PORT,
                    "resources": {
                        "limits": limits,
                        "requests": requests,
                    },
                },
                "distribution": {"name": distribution_name},
                "userConfig": {"configMapName": "llama-stack-config"},
            },
        },
    }


# =============================================================================
# VectorStore Ingestion Job
# =============================================================================

def build_vectorstore_job(
    workflow_name: str,
    namespace: str,
    workflow_spec: Dict[str, Any],
    workflow: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build ConfigMap and Job for vector store ingestion."""
    vector_store_spec = workflow_spec.get("vectorStore", {})
    
    if not vector_store_spec:
        return []
    
    store_name = vector_store_spec.get("name", "docs-vectorstore")
    embedding_model = vector_store_spec.get("embeddingModel", "granite-embedding-125m")
    chunk_size = vector_store_spec.get("chunkSize", 1000)
    chunk_overlap = vector_store_spec.get("chunkOverlap", 100)
    sources = vector_store_spec.get("sources", {})
    
    # Build list of URLs to ingest
    urls = sources.get("urls", [])
    
    # If GitHub source is specified, we'd need to expand it
    # For now, we support direct URLs
    
    if not urls:
        return []
    
    # Python script for ingestion using OpenAI client
    ingest_script = f'''#!/usr/bin/env python3
"""Vector store ingestion script using OpenAI client."""
import io
import os
import sys
import time
import requests
from openai import OpenAI

LLAMA_STACK_URL = os.environ.get("LLAMA_STACK_URL", "http://llama-stack-service:8321")
VECTOR_STORE_NAME = "{store_name}"
EMBEDDING_MODEL = "{embedding_model}"
CHUNK_SIZE = {chunk_size}
CHUNK_OVERLAP = {chunk_overlap}

URLS = {json.dumps(urls)}

# Initialize OpenAI client pointing to Llama Stack
client = OpenAI(base_url=LLAMA_STACK_URL, api_key="not-needed")

def fetch_and_upload_file(url):
    """Fetch and upload file via HTTP multipart (Llama Stack /v1/files)."""
    filename = url.split("/")[-1]
    print(f"Fetching: {{url}}")
    
    try:
        # Download content
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        content = resp.content
        
        # Upload using HTTP multipart to /v1/files
        print(f"Uploading: {{filename}}")
        files = {{"file": (filename, content, "application/octet-stream")}}
        data = {{"purpose": "assistants"}}
        upload_resp = requests.post(
            f"{{LLAMA_STACK_URL}}/v1/files",
            files=files,
            data=data,
            timeout=60
        )
        
        if upload_resp.status_code in [200, 201]:
            file_obj = upload_resp.json()
            file_id = file_obj.get("id")
            print(f"  ✓ Uploaded: {{filename}} (ID: {{file_id}})")
            return file_id
        else:
            print(f"  ✗ Upload failed: {{upload_resp.status_code}} - {{upload_resp.text}}")
            return None
    except Exception as e:
        print(f"  ✗ Failed: {{url}} - {{e}}")
        return None

def create_vector_store_with_files(file_ids):
    """Create vector store using direct HTTP (OpenAI client doesn't support Llama Stack vector stores)."""
    # Check if exists
    try:
        resp = requests.get(f"{{LLAMA_STACK_URL}}/v1/vector_stores", timeout=10)
        if resp.status_code == 200:
            for store in resp.json().get("data", []):
                if store.get("name") == VECTOR_STORE_NAME:
                    print(f"Vector store '{{VECTOR_STORE_NAME}}' exists: {{store['id']}}")
                    return store["id"]
    except:
        pass
    
    # Create new with files in one call
    payload = {{
        "name": VECTOR_STORE_NAME,
        "file_ids": file_ids,
        "chunking_strategy": {{
            "type": "static",
            "static": {{
                "max_chunk_size_tokens": CHUNK_SIZE,
                "chunk_overlap_tokens": CHUNK_OVERLAP,
            }}
        }},
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": 768,
        "provider_id": "milvus",
    }}
    
    resp = requests.post(f"{{LLAMA_STACK_URL}}/v1/vector_stores", json=payload, timeout=30)
    if resp.status_code in [200, 201]:
        store = resp.json()
        print(f"  ✓ Created vector store: {{store.get('id')}}")
        return store.get("id")
    else:
        print(f"Failed to create vector store: {{resp.status_code}} - {{resp.text}}")
        sys.exit(1)

def main():
    print("=" * 60)
    print("Vector Store Ingestion (OpenAI Client + HTTP)")
    print("=" * 60)
    print(f"Llama Stack: {{LLAMA_STACK_URL}}")
    print(f"Store: {{VECTOR_STORE_NAME}}")
    print(f"Embedding Model: {{EMBEDDING_MODEL}}")
    print(f"URLs: {{len(URLS)}}")
    print("=" * 60)
    
    # Upload all files using OpenAI client
    file_ids = []
    for url in URLS:
        file_id = fetch_and_upload_file(url)
        if file_id:
            file_ids.append(file_id)
    
    if not file_ids:
        print("No files uploaded!")
        sys.exit(1)
    
    print(f"\\nUploaded {{len(file_ids)}} files")
    
    # Warm up embedding model
    print("Warming up embedding model...")
    try:
        client.embeddings.create(model=EMBEDDING_MODEL, input=["warmup"])
        print("  ✓ Model ready")
    except Exception as e:
        print(f"Warning: {{e}}")
    
    # Create vector store with files
    print(f"\\nCreating vector store '{{VECTOR_STORE_NAME}}'...")
    store_id = create_vector_store_with_files(file_ids)
    
    # Wait for indexing
    print("\\nWaiting for indexing (30s)...")
    time.sleep(30)
    
    # Write store ID
    with open("/tmp/vectorstore-id.txt", "w") as f:
        f.write(store_id)
    
    print("=" * 60)
    print(f"✓ Done! Vector store ID: {{store_id}}")
    print("=" * 60)

if __name__ == "__main__":
    main()
'''
    
    # ConfigMap with the script
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": f"{workflow_name}-vectorstore-script",
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "vectorstore-script"),
            "ownerReferences": [build_owner_reference(workflow)],
        },
        "data": {
            "ingest.py": ingest_script,
        },
    }
    
    # Job to run the ingestion
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"{workflow_name}-vectorstore-setup",
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "vectorstore-job"),
            "ownerReferences": [build_owner_reference(workflow)],
        },
        "spec": {
            "ttlSecondsAfterFinished": 300,
            "backoffLimit": 3,
            "template": {
                "metadata": {
                    "labels": build_workflow_labels(workflow_name, "vectorstore-job"),
                },
                "spec": {
                    "restartPolicy": "OnFailure",
                    "containers": [{
                        "name": "setup",
                        "image": VECTORSTORE_SETUP_IMAGE,
                        "command": ["sh", "-c", "pip install --user requests openai && python /scripts/ingest.py"],
                        "env": [
                            {"name": "LLAMA_STACK_URL", "value": f"http://llama-stack-service.{namespace}.svc.cluster.local:{LLAMA_STACK_PORT}"},
                            {"name": "HOME", "value": "/tmp"},
                            {"name": "PYTHONUSERBASE", "value": "/tmp/.local"},
                            {"name": "PATH", "value": "/tmp/.local/bin:/usr/local/bin:/usr/bin:/bin"},
                        ],
                        "volumeMounts": [{
                            "name": "script",
                            "mountPath": "/scripts",
                        }],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "256Mi"},
                            "limits": {"cpu": "500m", "memory": "512Mi"},
                        },
                    }],
                    "volumes": [{
                        "name": "script",
                        "configMap": {"name": f"{workflow_name}-vectorstore-script"},
                    }],
                },
            },
        },
    }
    
    return [configmap, job]


# =============================================================================
# OTEL Collector Resources
# =============================================================================

def build_otel_collector(
    workflow_name: str,
    namespace: str,
    workflow_spec: Dict[str, Any],
    workflow: Dict[str, Any],
    platform_namespace: str = "rh-agentic-system",
) -> List[Dict[str, Any]]:
    """Build OTEL Collector Deployment, ConfigMap, and Service.
    
    MLflow 3.x+ supports native OTLP trace ingestion at /v1/traces endpoint.
    We configure the OTEL collector to export directly to MLflow.
    """
    observability = workflow_spec.get("observability", {})
    
    if not observability.get("otelEnabled", True):
        return []
    
    # Get experiment name for MLflow header
    experiment_name = observability.get("mlflowExperiment", workflow_name)
    
    # MLflow 3.x+ native OTLP endpoint - traces go directly to MLflow
    # Note: OTLP HTTP exporter auto-appends /v1/traces, so we just use base URL
    mlflow_base_url = f"http://mlflow.{platform_namespace}.svc.cluster.local:5000"
    
    config_yaml = f'''extensions:
  health_check:
    endpoint: 0.0.0.0:13133

receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:{OTEL_GRPC_PORT}
      http:
        endpoint: 0.0.0.0:{OTEL_HTTP_PORT}

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
  memory_limiter:
    check_interval: 1s
    limit_mib: 512
    spike_limit_mib: 128

exporters:
  # Export directly to MLflow 3.x native OTLP endpoint
  otlphttp/mlflow:
    endpoint: {mlflow_base_url}
    tls:
      insecure: true
    headers:
      # MLflow 3.x requires experiment ID header for traces
      x-mlflow-experiment-id: "0"
  debug:
    verbosity: detailed

service:
  extensions: [health_check]
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlphttp/mlflow, debug]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [debug]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [debug]
'''
    
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "otel-collector-config",
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "otel-config"),
            "ownerReferences": [build_owner_reference(workflow)],
        },
        "data": {
            "config.yaml": config_yaml,
        },
    }
    
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "otel-collector",
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "otel-collector"),
            "ownerReferences": [build_owner_reference(workflow)],
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": "otel-collector",
                    "agents.redhat.com/workflow": workflow_name,
                },
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "otel-collector",
                        "agents.redhat.com/workflow": workflow_name,
                    },
                },
                "spec": {
                    "containers": [{
                        "name": "otel-collector",
                        "image": OTEL_COLLECTOR_IMAGE,
                        "args": ["--config=/etc/otel/config.yaml"],
                        "ports": [
                            {"containerPort": OTEL_GRPC_PORT, "name": "otlp-grpc", "protocol": "TCP"},
                            {"containerPort": OTEL_HTTP_PORT, "name": "otlp-http", "protocol": "TCP"},
                        ],
                        "volumeMounts": [{
                            "name": "config",
                            "mountPath": "/etc/otel",
                        }],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "256Mi"},
                            "limits": {"cpu": "500m", "memory": "512Mi"},
                        },
                        "livenessProbe": {
                            "httpGet": {"path": "/", "port": 13133},
                            "initialDelaySeconds": 10,
                            "periodSeconds": 10,
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/", "port": 13133},
                            "initialDelaySeconds": 5,
                            "periodSeconds": 5,
                        },
                    }],
                    "volumes": [{
                        "name": "config",
                        "configMap": {"name": "otel-collector-config"},
                    }],
                },
            },
        },
    }
    
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "otel-collector",
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "otel-service"),
            "ownerReferences": [build_owner_reference(workflow)],
        },
        "spec": {
            "selector": {
                "app": "otel-collector",
                "agents.redhat.com/workflow": workflow_name,
            },
            "ports": [
                {"name": "otlp-grpc", "port": OTEL_GRPC_PORT, "targetPort": OTEL_GRPC_PORT},
                {"name": "otlp-http", "port": OTEL_HTTP_PORT, "targetPort": OTEL_HTTP_PORT},
            ],
        },
    }
    
    return [configmap, deployment, service]


# =============================================================================
# MLflow OTEL Bridge
# =============================================================================

def build_mlflow_otel_bridge(
    workflow_name: str,
    namespace: str,
    workflow_spec: Dict[str, Any],
    workflow: Dict[str, Any],
    platform_namespace: str = "rh-agentic-system",
) -> List[Dict[str, Any]]:
    """Build MLflow OTEL Bridge that converts OTLP spans to MLflow traces.
    
    NOTE: This bridge is DEPRECATED with MLflow 3.x+ which has native OTLP support.
    The OTEL collector now sends traces directly to MLflow's /v1/traces endpoint.
    This bridge is kept for backward compatibility but disabled by default.
    Set observability.useLegacyBridge: true to enable.
    """
    observability = workflow_spec.get("observability", {})
    
    # Disabled by default - MLflow 3.x has native OTLP support
    if not observability.get("useLegacyBridge", False):
        return []
    
    # Python script for the bridge
    bridge_script = '''#!/usr/bin/env python3
"""MLflow OTEL Bridge - Converts OTLP spans to MLflow traces.

This is a simple bridge that:
1. Receives OTLP spans via HTTP
2. Converts them to MLflow trace format
3. Logs them to MLflow tracking server
"""
import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify

import mlflow
from mlflow.entities import SpanType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "llama-stack-traces")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

@app.route("/v1/traces", methods=["POST"])
def receive_traces():
    """Receive OTLP traces and convert to MLflow."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "no data"}), 200
        
        resource_spans = data.get("resourceSpans", [])
        span_count = 0
        
        for rs in resource_spans:
            for scope_spans in rs.get("scopeSpans", []):
                for span in scope_spans.get("spans", []):
                    try:
                        _log_span_to_mlflow(span)
                        span_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to log span: {e}")
        
        logger.info(f"Processed {span_count} spans")
        return jsonify({"status": "ok", "spans_processed": span_count}), 200
        
    except Exception as e:
        logger.error(f"Error processing traces: {e}")
        return jsonify({"error": str(e)}), 500


def _log_span_to_mlflow(span: dict):
    """Convert OTLP span to MLflow trace."""
    name = span.get("name", "unknown")
    trace_id = span.get("traceId", "")
    span_id = span.get("spanId", "")
    
    # Extract attributes
    attributes = {}
    for attr in span.get("attributes", []):
        key = attr.get("key", "")
        value = attr.get("value", {})
        if "stringValue" in value:
            attributes[key] = value["stringValue"]
        elif "intValue" in value:
            attributes[key] = int(value["intValue"])
        elif "boolValue" in value:
            attributes[key] = value["boolValue"]
    
    # Log as MLflow trace using the tracing API
    with mlflow.start_span(name=name) as mlflow_span:
        mlflow_span.set_attributes(attributes)
        mlflow_span.set_attribute("otel.trace_id", trace_id)
        mlflow_span.set_attribute("otel.span_id", span_id)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    logger.info(f"Starting MLflow OTEL Bridge")
    logger.info(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Experiment: {MLFLOW_EXPERIMENT}")
    app.run(host="0.0.0.0", port=4318)
'''
    
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "mlflow-otel-bridge-script",
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "mlflow-bridge-script"),
            "ownerReferences": [build_owner_reference(workflow)],
        },
        "data": {
            "bridge.py": bridge_script,
            "requirements.txt": "flask\nmlflow>=3.0.0\nopentelemetry-api\nopentelemetry-sdk\nboto3\n",
        },
    }
    
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "mlflow-otel-bridge",
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "mlflow-bridge"),
            "ownerReferences": [build_owner_reference(workflow)],
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": "mlflow-otel-bridge",
                    "agents.redhat.com/workflow": workflow_name,
                },
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "mlflow-otel-bridge",
                        "agents.redhat.com/workflow": workflow_name,
                    },
                },
                "spec": {
                    "containers": [{
                        "name": "bridge",
                        "image": MLFLOW_OTEL_BRIDGE_IMAGE,
                        "command": ["sh", "-c", "pip install -r /app/requirements.txt && python /app/bridge.py"],
                        "ports": [{"containerPort": OTEL_HTTP_PORT, "name": "http"}],
                        "env": [
                            {"name": "HOME", "value": "/tmp"},
                            {"name": "MLFLOW_TRACKING_URI", "value": f"http://mlflow.{platform_namespace}.svc.cluster.local:5000"},
                            {"name": "MLFLOW_EXPERIMENT", "value": workflow_name},
                            # MLflow credentials from platform secret
                            {
                                "name": "AWS_ACCESS_KEY_ID",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "mlflow-credentials",
                                        "key": "AWS_ACCESS_KEY_ID",
                                        "optional": True,
                                    }
                                }
                            },
                            {
                                "name": "AWS_SECRET_ACCESS_KEY",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "mlflow-credentials",
                                        "key": "AWS_SECRET_ACCESS_KEY",
                                        "optional": True,
                                    }
                                }
                            },
                            {"name": "MLFLOW_S3_ENDPOINT_URL", "value": f"http://mlflow-minio.{platform_namespace}.svc.cluster.local:9000"},
                        ],
                        "volumeMounts": [{
                            "name": "script",
                            "mountPath": "/app",
                        }],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "256Mi"},
                            "limits": {"cpu": "500m", "memory": "512Mi"},
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/health", "port": OTEL_HTTP_PORT},
                            "initialDelaySeconds": 10,
                            "periodSeconds": 10,
                        },
                    }],
                    "volumes": [{
                        "name": "script",
                        "configMap": {"name": "mlflow-otel-bridge-script"},
                    }],
                },
            },
        },
    }
    
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "mlflow-otel-bridge",
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "mlflow-bridge-service"),
            "ownerReferences": [build_owner_reference(workflow)],
        },
        "spec": {
            "selector": {
                "app": "mlflow-otel-bridge",
                "agents.redhat.com/workflow": workflow_name,
            },
            "ports": [
                {"name": "http", "port": OTEL_HTTP_PORT, "targetPort": OTEL_HTTP_PORT},
            ],
        },
    }
    
    return [configmap, deployment, service]


# =============================================================================
# vLLM BuildConfig (In-Cluster Image Build)
# =============================================================================

def build_vllm_buildconfig(
    workflow_name: str,
    namespace: str,
    workflow_spec: Dict[str, Any],
    workflow: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build ImageStream and BuildConfig for vLLM with OTEL support."""
    model_spec = workflow_spec.get("model", {})
    vllm_config = model_spec.get("vllm", {})
    
    if not vllm_config.get("buildInCluster", True):
        return []
    
    # Dockerfile for vLLM with OTEL
    dockerfile = '''FROM vllm/vllm-openai:latest

# Install OpenTelemetry dependencies
RUN pip install --no-cache-dir \\
    opentelemetry-api \\
    opentelemetry-sdk \\
    opentelemetry-exporter-otlp \\
    opentelemetry-instrumentation

# Set default OTEL environment
ENV OTEL_TRACES_EXPORTER=otlp
ENV OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
'''
    
    # ImageStream
    imagestream = {
        "apiVersion": "image.openshift.io/v1",
        "kind": "ImageStream",
        "metadata": {
            "name": "vllm-openai-otel",
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "vllm-imagestream"),
            "ownerReferences": [build_owner_reference(workflow)],
        },
    }
    
    # BuildConfig
    buildconfig = {
        "apiVersion": "build.openshift.io/v1",
        "kind": "BuildConfig",
        "metadata": {
            "name": "vllm-openai-otel",
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, "vllm-buildconfig"),
            "ownerReferences": [build_owner_reference(workflow)],
        },
        "spec": {
            "source": {
                "type": "Dockerfile",
                "dockerfile": dockerfile,
            },
            "strategy": {
                "type": "Docker",
                "dockerStrategy": {
                    "from": {
                        "kind": "DockerImage",
                        "name": "vllm/vllm-openai:latest",
                    },
                },
            },
            "output": {
                "to": {
                    "kind": "ImageStreamTag",
                    "name": "vllm-openai-otel:latest",
                },
            },
            "triggers": [
                {"type": "ConfigChange"},
            ],
        },
    }
    
    return [imagestream, buildconfig]


# =============================================================================
# BaseAgent CR Builder (from AgentWorkflow agents spec)
# =============================================================================

def build_baseagent_from_workflow(
    workflow_name: str,
    namespace: str,
    agent_spec: Dict[str, Any],
    workflow_spec: Dict[str, Any],
    workflow: Dict[str, Any],
    vector_store_id: Optional[str] = None,
    platform_namespace: str = "rh-agentic-system",
) -> Dict[str, Any]:
    """Build a BaseAgent CR from AgentWorkflow agent spec."""
    agent_name = agent_spec["name"]
    agent_type = agent_spec.get("type", "simple")
    instruction = agent_spec.get("instruction", "")
    description = agent_spec.get("description", f"{agent_name} agent")
    
    model_spec = workflow_spec.get("model", {})
    model_id = model_spec.get("modelId", model_spec.get("name", "qwen3-14b-awq"))
    
    # Base spec
    spec = {
        "instruction": instruction,
        "openai": {
            "baseUrl": f"http://llama-stack-service.{namespace}.svc.cluster.local:{LLAMA_STACK_PORT}",
            "model": f"vllm-inference-1/{model_id}",
        },
        "agentCard": {
            "description": description,
        },
        "kagenti": {
            "enabled": True,
        },
        "mlflow": {
            "trackingUri": f"http://mlflow.{platform_namespace}.svc.cluster.local:5000",
            "experiment": agent_name,
            "s3EndpointUrl": f"http://mlflow-minio.{platform_namespace}.svc.cluster.local:9000",
            "credentialsSecretRef": {
                "name": "mlflow-credentials",
            },
        },
    }
    
    # Add RAG config
    if agent_type == "rag":
        rag_spec = agent_spec.get("rag", {})
        if rag_spec.get("vectorStoreFromWorkflow", True) and vector_store_id:
            spec["rag"] = {
                "vectorStoreIds": [vector_store_id],
                "maxResults": rag_spec.get("maxResults", 10),
            }
        elif rag_spec.get("vectorStoreIds"):
            spec["rag"] = {
                "vectorStoreIds": rag_spec["vectorStoreIds"],
                "maxResults": rag_spec.get("maxResults", 10),
            }
    
    # Add MCP tools config
    if agent_type == "mcp":
        mcp_tool_names = agent_spec.get("mcpTools", [])
        workflow_mcp_tools = workflow_spec.get("mcpTools", [])
        
        mcp_tools = []
        for tool_name in mcp_tool_names:
            for wf_tool in workflow_mcp_tools:
                if wf_tool.get("name") == tool_name:
                    tool_config = {
                        "serverUrl": wf_tool.get("serverUrl", ""),
                        "serverLabel": tool_name,
                    }
                    if wf_tool.get("secretRef"):
                        tool_config["secretRef"] = wf_tool["secretRef"]
                    mcp_tools.append(tool_config)
                    break
        
        if mcp_tools:
            spec["mcpTools"] = mcp_tools
    
    # Add subagents config for orchestrator
    if agent_type == "orchestrator":
        subagent_names = agent_spec.get("subagents", [])
        subagents = []
        for sub_name in subagent_names:
            subagents.append({
                "name": sub_name,
                "url": f"http://{sub_name}.{namespace}.svc.cluster.local:{C.ADAPTER_PORT}",
                "skill": "answer",
                "batch": "default",
            })
        if subagents:
            spec["subagents"] = subagents
    
    return {
        "apiVersion": f"{C.API_GROUP}/{C.API_VERSION}",
        "kind": "BaseAgent",
        "metadata": {
            "name": agent_name,
            "namespace": namespace,
            "labels": build_workflow_labels(workflow_name, f"agent-{agent_type}"),
            "ownerReferences": [build_owner_reference(workflow)],
        },
        "spec": spec,
    }

