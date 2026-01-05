"""Default values and constants for rh-agentic-operator."""

import os

# API Group and Version
API_GROUP = "agents.redhat.com"
API_VERSION = "v1alpha1"
PLURAL = "baseagents"

# Operator name
OPERATOR_NAME = "rh-agentic-operator"

# =============================================================================
# Component Images (configurable via environment variables)
# =============================================================================

# Base agent image (the adapter that wraps LLM calls)
DEFAULT_BASE_AGENT_IMAGE = os.getenv(
    "BASE_AGENT_IMAGE",
    "quay.io/opendatahub/base-agent:latest"
)

# MLflow images
MLFLOW_IMAGE = os.getenv(
    "MLFLOW_IMAGE",
    "ghcr.io/mlflow/mlflow:v2.18.0"
)
MLFLOW_MINIO_IMAGE = os.getenv(
    "MLFLOW_MINIO_IMAGE",
    "quay.io/minio/minio:latest"
)
MLFLOW_POSTGRES_IMAGE = os.getenv(
    "MLFLOW_POSTGRES_IMAGE",
    "postgres:15"
)

# Kagenti images
KAGENTI_UI_IMAGE = os.getenv(
    "KAGENTI_UI_IMAGE",
    "ghcr.io/kagenti/kagenti/ui:v0.2.0-alpha.2"
)
KAGENTI_PROXY_IMAGE = os.getenv(
    "KAGENTI_PROXY_IMAGE",
    "nginx:1.25-alpine"
)

# Kagent images (if operator deploys kagent controller)
KAGENT_CONTROLLER_IMAGE = os.getenv(
    "KAGENT_CONTROLLER_IMAGE",
    "cr.kagent.dev/kagent-dev/kagent/controller:0.7.7"
)

# =============================================================================
# Default OpenAI-Compatible Endpoint Configuration
# =============================================================================

DEFAULT_OPENAI_BASE_URL = os.getenv(
    "DEFAULT_OPENAI_BASE_URL",
    "http://llama-stack-service:8321"
)
DEFAULT_OPENAI_MODEL = os.getenv(
    "DEFAULT_OPENAI_MODEL",
    "vllm-inference-1/qwen3-14b-awq"
)

# =============================================================================
# Default RAG Configuration
# =============================================================================

DEFAULT_MAX_RESULTS = int(os.getenv("DEFAULT_MAX_RESULTS", "10"))

# =============================================================================
# Default Orchestration Configuration
# =============================================================================

DEFAULT_A2A_MAX_SELECTED = int(os.getenv("DEFAULT_A2A_MAX_SELECTED", "4"))
DEFAULT_A2A_CALL_TIMEOUT_S = int(os.getenv("DEFAULT_A2A_CALL_TIMEOUT_S", "60"))
DEFAULT_A2A_DEFAULT_BATCH = os.getenv("DEFAULT_A2A_DEFAULT_BATCH", "default")

# =============================================================================
# Default Resources
# =============================================================================

DEFAULT_CPU_REQUEST = os.getenv("DEFAULT_CPU_REQUEST", "100m")
DEFAULT_MEMORY_REQUEST = os.getenv("DEFAULT_MEMORY_REQUEST", "256Mi")
DEFAULT_CPU_LIMIT = os.getenv("DEFAULT_CPU_LIMIT", "500m")
DEFAULT_MEMORY_LIMIT = os.getenv("DEFAULT_MEMORY_LIMIT", "1Gi")

# =============================================================================
# Ports
# =============================================================================

KAGENTI_PROXY_PORT = 8000
ADAPTER_PORT = 8080
MLFLOW_PORT = 5000
MINIO_API_PORT = 9000
MINIO_CONSOLE_PORT = 9001
POSTGRES_PORT = 5432

# =============================================================================
# Labels
# =============================================================================

LABEL_MANAGED_BY = "app.kubernetes.io/managed-by"
LABEL_INSTANCE = "app.kubernetes.io/instance"
LABEL_COMPONENT = "app.kubernetes.io/component"
