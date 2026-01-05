"""
Platform deployment module - deploys MLflow, kagent, kagenti in namespace.
"""
import secrets
import string
import logging
from typing import Dict, Any, List

from . import constants as C

logger = logging.getLogger(__name__)


def generate_password(length: int = 24) -> str:
    """Generate a secure random password."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def build_platform_labels(name: str, component: str) -> Dict[str, str]:
    """Build labels for platform components."""
    return {
        "app.kubernetes.io/name": name,
        "app.kubernetes.io/component": component,
        "app.kubernetes.io/managed-by": "rh-agentic-operator",
        "app.kubernetes.io/part-of": "agentic-platform",
    }


def get_image(spec: Dict[str, Any], component: str, default: str) -> str:
    """Get image from spec override or use default from constants."""
    # Check images section first
    images = spec.get("images", {})
    if component in images and images[component]:
        return images[component]
    # Check component-specific override
    components = spec.get("components", {})
    if component == "mlflow" and "mlflow" in components:
        if components["mlflow"].get("image"):
            return components["mlflow"]["image"]
    elif component == "minio" and "mlflow" in components:
        if components["mlflow"].get("minioImage"):
            return components["mlflow"]["minioImage"]
    elif component == "postgres" and "mlflow" in components:
        if components["mlflow"].get("postgresImage"):
            return components["mlflow"]["postgresImage"]
    elif component == "kagentiUI" and "kagenti" in components:
        if components["kagenti"].get("image"):
            return components["kagenti"]["image"]
    return default


def build_mlflow_resources(namespace: str, spec: Dict[str, Any], owner: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build all MLflow-related resources."""
    mlflow_spec = spec.get("components", {}).get("mlflow", {})
    storage_size = mlflow_spec.get("storage", {}).get("size", "10Gi")
    postgres_storage = mlflow_spec.get("postgres", {}).get("storage", "5Gi")
    storage_class = mlflow_spec.get("storage", {}).get("storageClassName")
    
    # Get images (from spec or defaults)
    mlflow_image = get_image(spec, "mlflow", C.MLFLOW_IMAGE)
    minio_image = get_image(spec, "minio", C.MLFLOW_MINIO_IMAGE)
    postgres_image = get_image(spec, "postgres", C.MLFLOW_POSTGRES_IMAGE)
    
    # Use fixed credentials for internal services
    # These are internal-only services not exposed outside the cluster
    minio_user = "mlflow"
    minio_password = "mlflowaccesskey"  # Fixed password for MinIO (internal only)
    postgres_password = "mlflowdbpass"    # Fixed password for PostgreSQL (internal only)
    
    owner_ref = {
        "apiVersion": f"{C.API_GROUP}/{C.API_VERSION}",
        "kind": "AgenticPlatform",
        "name": owner["metadata"]["name"],
        "uid": owner["metadata"]["uid"],
        "controller": True,
    }
    
    resources = []
    
    # 1. Credentials Secret
    resources.append({
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": "mlflow-credentials",
            "namespace": namespace,
            "labels": build_platform_labels("mlflow", "credentials"),
            "ownerReferences": [owner_ref],
        },
        "type": "Opaque",
        "stringData": {
            "AWS_ACCESS_KEY_ID": minio_user,
            "AWS_SECRET_ACCESS_KEY": minio_password,
            "MINIO_ROOT_USER": minio_user,
            "MINIO_ROOT_PASSWORD": minio_password,
            "POSTGRES_USER": "mlflow",
            "POSTGRES_PASSWORD": postgres_password,
            "POSTGRES_DB": "mlflow",
        },
    })
    
    # 2. MinIO PVC
    minio_pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": "mlflow-minio-pvc",
            "namespace": namespace,
            "labels": build_platform_labels("mlflow", "minio"),
            "ownerReferences": [owner_ref],
        },
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": storage_size}},
        },
    }
    if storage_class:
        minio_pvc["spec"]["storageClassName"] = storage_class
    resources.append(minio_pvc)
    
    # 3. Postgres PVC
    postgres_pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": "mlflow-postgres-pvc",
            "namespace": namespace,
            "labels": build_platform_labels("mlflow", "postgres"),
            "ownerReferences": [owner_ref],
        },
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": postgres_storage}},
        },
    }
    if storage_class:
        postgres_pvc["spec"]["storageClassName"] = storage_class
    resources.append(postgres_pvc)
    
    # 4. MinIO Deployment
    resources.append({
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "mlflow-minio",
            "namespace": namespace,
            "labels": build_platform_labels("mlflow", "minio"),
            "ownerReferences": [owner_ref],
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "mlflow-minio"}},
            "template": {
                "metadata": {"labels": {"app": "mlflow-minio"}},
                "spec": {
                    "containers": [{
                        "name": "minio",
                        "image": minio_image,
                        "args": ["server", "/data", "--console-address", ":9001"],
                        "ports": [
                            {"containerPort": 9000, "name": "api"},
                            {"containerPort": 9001, "name": "console"},
                        ],
                        "envFrom": [{"secretRef": {"name": "mlflow-credentials"}}],
                        "volumeMounts": [{"name": "data", "mountPath": "/data"}],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "256Mi"},
                            "limits": {"cpu": "500m", "memory": "512Mi"},
                        },
                    }],
                    "volumes": [{"name": "data", "persistentVolumeClaim": {"claimName": "mlflow-minio-pvc"}}],
                },
            },
        },
    })
    
    # 5. MinIO Service
    resources.append({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "mlflow-minio",
            "namespace": namespace,
            "labels": build_platform_labels("mlflow", "minio"),
            "ownerReferences": [owner_ref],
        },
        "spec": {
            "selector": {"app": "mlflow-minio"},
            "ports": [
                {"port": 9000, "targetPort": 9000, "name": "api"},
                {"port": 9001, "targetPort": 9001, "name": "console"},
            ],
        },
    })
    
    # 6. Postgres Deployment
    resources.append({
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "mlflow-postgres",
            "namespace": namespace,
            "labels": build_platform_labels("mlflow", "postgres"),
            "ownerReferences": [owner_ref],
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "mlflow-postgres"}},
            "template": {
                "metadata": {"labels": {"app": "mlflow-postgres"}},
                "spec": {
                    "containers": [{
                        "name": "postgres",
                        "image": postgres_image,
                        "ports": [{"containerPort": 5432}],
                        "envFrom": [{"secretRef": {"name": "mlflow-credentials"}}],
                        "env": [
                            {"name": "PGDATA", "value": "/var/lib/postgresql/data/pgdata"},
                        ],
                        "volumeMounts": [{"name": "data", "mountPath": "/var/lib/postgresql/data"}],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "256Mi"},
                            "limits": {"cpu": "500m", "memory": "512Mi"},
                        },
                    }],
                    "volumes": [{"name": "data", "persistentVolumeClaim": {"claimName": "mlflow-postgres-pvc"}}],
                },
            },
        },
    })
    
    # 7. Postgres Service
    resources.append({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "mlflow-postgres",
            "namespace": namespace,
            "labels": build_platform_labels("mlflow", "postgres"),
            "ownerReferences": [owner_ref],
        },
        "spec": {
            "selector": {"app": "mlflow-postgres"},
            "ports": [{"port": 5432, "targetPort": 5432}],
        },
    })
    
    # 8. MLflow Deployment
    resources.append({
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "mlflow",
            "namespace": namespace,
            "labels": build_platform_labels("mlflow", "server"),
            "ownerReferences": [owner_ref],
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "mlflow"}},
            "template": {
                "metadata": {"labels": {"app": "mlflow"}},
                "spec": {
                    "initContainers": [{
                        "name": "wait-for-postgres",
                        "image": "busybox:1.36",
                        "command": ["sh", "-c", 
                            "until nc -z mlflow-postgres 5432; do echo waiting for postgres; sleep 2; done"],
                    }],
                    "containers": [{
                        "name": "mlflow",
                        "image": mlflow_image,
                        "command": ["/bin/bash", "-c"],
                        "args": [
                            "pip install --user --quiet psycopg2-binary boto3 && "
                            "mlflow server --host 0.0.0.0 --port 5000 "
                            f"--backend-store-uri 'postgresql://mlflow:$(POSTGRES_PASSWORD)@mlflow-postgres:5432/mlflow' "
                            "--default-artifact-root 's3://mlflow/'"
                        ],
                        "ports": [{"containerPort": 5000}],
                        "env": [
                            {"name": "POSTGRES_PASSWORD", "valueFrom": {
                                "secretKeyRef": {"name": "mlflow-credentials", "key": "POSTGRES_PASSWORD"}}},
                            {"name": "AWS_ACCESS_KEY_ID", "valueFrom": {
                                "secretKeyRef": {"name": "mlflow-credentials", "key": "AWS_ACCESS_KEY_ID"}}},
                            {"name": "AWS_SECRET_ACCESS_KEY", "valueFrom": {
                                "secretKeyRef": {"name": "mlflow-credentials", "key": "AWS_SECRET_ACCESS_KEY"}}},
                            {"name": "MLFLOW_S3_ENDPOINT_URL", "value": f"http://mlflow-minio.{namespace}.svc.cluster.local:9000"},
                            {"name": "HOME", "value": "/tmp"},
                            {"name": "PIP_CACHE_DIR", "value": "/tmp/.pip-cache"},
                            {"name": "PYTHONUSERBASE", "value": "/tmp/.local"},
                            {"name": "PATH", "value": "/tmp/.local/bin:/usr/local/bin:/usr/bin:/bin"},
                        ],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "512Mi"},
                            "limits": {"cpu": "1000m", "memory": "1Gi"},
                        },
                    }],
                },
            },
        },
    })
    
    # 9. MLflow Service
    resources.append({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "mlflow",
            "namespace": namespace,
            "labels": build_platform_labels("mlflow", "server"),
            "ownerReferences": [owner_ref],
        },
        "spec": {
            "selector": {"app": "mlflow"},
            "ports": [{"port": 5000, "targetPort": 5000}],
        },
    })
    
    # 10. MLflow Route (OpenShift)
    resources.append({
        "apiVersion": "route.openshift.io/v1",
        "kind": "Route",
        "metadata": {
            "name": "mlflow",
            "namespace": namespace,
            "labels": build_platform_labels("mlflow", "route"),
            "ownerReferences": [owner_ref],
        },
        "spec": {
            "to": {"kind": "Service", "name": "mlflow"},
            "port": {"targetPort": 5000},
            "tls": {"termination": "edge", "insecureEdgeTerminationPolicy": "Redirect"},
        },
    })
    
    return resources


def build_kagenti_resources(namespace: str, spec: Dict[str, Any], owner: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build Kagenti UI deployment resources."""
    owner_ref = {
        "apiVersion": f"{C.API_GROUP}/{C.API_VERSION}",
        "kind": "AgenticPlatform",
        "name": owner["metadata"]["name"],
        "uid": owner["metadata"]["uid"],
        "controller": True,
    }
    
    # Get kagenti image (from spec or default)
    kagenti_image = get_image(spec, "kagentiUI", C.KAGENTI_UI_IMAGE)
    
    resources = []
    
    # 1. Kagenti Service Account
    resources.append({
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {
            "name": "kagenti-ui",
            "namespace": namespace,
            "labels": build_platform_labels("kagenti", "serviceaccount"),
            "ownerReferences": [owner_ref],
        },
    })
    
    # 2. Kagenti Role (to list agents in namespace)
    resources.append({
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {
            "name": "kagenti-ui",
            "namespace": namespace,
            "labels": build_platform_labels("kagenti", "role"),
            "ownerReferences": [owner_ref],
        },
        "rules": [
            {
                "apiGroups": ["agent.kagenti.dev"],
                "resources": ["agents", "agentcards", "agentbuilds"],
                "verbs": ["get", "list", "watch"],
            },
            {
                "apiGroups": ["kagent.dev"],
                "resources": ["agents", "modelconfigs"],
                "verbs": ["get", "list", "watch"],
            },
        ],
    })
    
    # 3. Kagenti RoleBinding
    resources.append({
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {
            "name": "kagenti-ui",
            "namespace": namespace,
            "labels": build_platform_labels("kagenti", "rolebinding"),
            "ownerReferences": [owner_ref],
        },
        "subjects": [{
            "kind": "ServiceAccount",
            "name": "kagenti-ui",
            "namespace": namespace,
        }],
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": "kagenti-ui",
        },
    })
    
    # 4. Kagenti UI Deployment
    resources.append({
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "kagenti-ui",
            "namespace": namespace,
            "labels": build_platform_labels("kagenti", "ui"),
            "ownerReferences": [owner_ref],
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "kagenti-ui"}},
            "template": {
                "metadata": {"labels": {"app": "kagenti-ui"}},
                "spec": {
                    "serviceAccountName": "kagenti-ui",
                    "containers": [{
                        "name": "kagenti-ui",
                        "image": kagenti_image,
                        "ports": [{"containerPort": 8501}],
                        "env": [
                            {"name": "NAMESPACE", "value": namespace},
                            {"name": "DEFAULT_NAMESPACE", "value": namespace},
                            {"name": "HOME", "value": "/tmp"},
                            {"name": "UV_CACHE_DIR", "value": "/tmp/.uv-cache"},
                            {"name": "XDG_CACHE_HOME", "value": "/tmp/.cache"},
                        ],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "256Mi"},
                            "limits": {"cpu": "500m", "memory": "512Mi"},
                        },
                    }],
                },
            },
        },
    })
    
    # 2. Kagenti Service (Streamlit runs on 8501)
    resources.append({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "kagenti-ui",
            "namespace": namespace,
            "labels": build_platform_labels("kagenti", "ui"),
            "ownerReferences": [owner_ref],
        },
        "spec": {
            "selector": {"app": "kagenti-ui"},
            "ports": [{"port": 3000, "targetPort": 8501}],
        },
    })
    
    # 3. Kagenti Route (targetPort must be the actual container port)
    resources.append({
        "apiVersion": "route.openshift.io/v1",
        "kind": "Route",
        "metadata": {
            "name": "kagenti-ui",
            "namespace": namespace,
            "labels": build_platform_labels("kagenti", "route"),
            "ownerReferences": [owner_ref],
        },
        "spec": {
            "to": {"kind": "Service", "name": "kagenti-ui"},
            "port": {"targetPort": 8501},
            "tls": {"termination": "edge", "insecureEdgeTerminationPolicy": "Redirect"},
        },
    })
    
    return resources


def get_mlflow_config_for_namespace(namespace: str) -> Dict[str, Any]:
    """Get the auto-generated MLflow config for agents in this namespace."""
    return {
        "trackingUri": f"http://mlflow.{namespace}.svc.cluster.local:5000",
        "s3EndpointUrl": f"http://mlflow-minio.{namespace}.svc.cluster.local:9000",
        "credentialsSecretRef": {
            "name": "mlflow-credentials",
            "accessKeyIdKey": "AWS_ACCESS_KEY_ID",
            "secretAccessKeyKey": "AWS_SECRET_ACCESS_KEY",
        },
    }


def get_platform_defaults(namespace: str, platform_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Get default values that should be applied to all BaseAgents in this namespace.
    
    Note: OpenAI endpoint is external (OpenAI, Llama Stack, vLLM, etc.) - we only provide defaults, not deploy it.
    """
    defaults = platform_spec.get("defaults", {})
    
    return {
        # OpenAI-compatible endpoint defaults (external service - agents can override)
        "openai": platform_spec.get("openaiDefaults", {}),
        # MLflow config (deployed by platform if enabled)
        "mlflow": get_mlflow_config_for_namespace(namespace),
        # Kagenti integration
        "kagenti": defaults.get("kagenti", {"enabled": True}),
        "replicas": defaults.get("replicas", 1),
        "resources": defaults.get("resources", {}),
    }


def check_kagent_installed(custom_api) -> bool:
    """Check if kagent CRDs are installed in the cluster.
    
    Returns True if the kagent.dev Agent CRD exists.
    """
    from kubernetes.client.rest import ApiException
    from kubernetes import client
    
    try:
        # Try to list kagent CRDs
        api_ext = client.ApiextensionsV1Api()
        crd = api_ext.read_custom_resource_definition("agents.kagent.dev")
        if crd:
            logger.info("Kagent CRD 'agents.kagent.dev' found - kagent is installed")
            return True
    except ApiException as e:
        if e.status == 404:
            logger.info("Kagent CRD not found - kagent is NOT installed")
            return False
        logger.warning(f"Error checking kagent CRD: {e}")
    return False


def build_kagenti_cluster_rbac(namespace: str, owner: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build ClusterRole and ClusterRoleBinding for Kagenti UI to read agents across all namespaces.
    
    The kagenti-ui ServiceAccount in the operator namespace gets cluster-wide read access.
    """
    owner_ref = {
        "apiVersion": f"{C.API_GROUP}/{C.API_VERSION}",
        "kind": "AgenticPlatform",
        "name": owner["metadata"]["name"],
        "uid": owner["metadata"]["uid"],
        "controller": True,
    }
    
    resources = []
    
    # ClusterRole for reading agents cluster-wide
    resources.append({
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "ClusterRole",
        "metadata": {
            "name": f"kagenti-agent-reader-{namespace}",
            "labels": build_platform_labels("kagenti", "cluster-role"),
            # Note: ClusterRoles cannot have ownerReferences to namespaced resources
            # So we use labels for tracking instead
        },
        "rules": [
            {
                "apiGroups": ["agent.kagenti.dev"],
                "resources": ["agents", "agentcards", "agentbuilds"],
                "verbs": ["get", "list", "watch"],
            },
            {
                "apiGroups": ["kagent.dev"],
                "resources": ["agents", "modelconfigs"],
                "verbs": ["get", "list", "watch"],
            },
            {
                "apiGroups": [""],
                "resources": ["namespaces"],
                "verbs": ["get", "list", "watch"],
            },
        ],
    })
    
    # ClusterRoleBinding to bind the ClusterRole to kagenti-ui ServiceAccount
    resources.append({
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "ClusterRoleBinding",
        "metadata": {
            "name": f"kagenti-agent-reader-{namespace}",
            "labels": build_platform_labels("kagenti", "cluster-rolebinding"),
        },
        "subjects": [{
            "kind": "ServiceAccount",
            "name": "kagenti-ui",
            "namespace": namespace,
        }],
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "ClusterRole",
            "name": f"kagenti-agent-reader-{namespace}",
        },
    })
    
    return resources


def build_mlflow_credentials_copy(source_namespace: str, target_namespace: str) -> Dict[str, Any]:
    """Build a Secret that copies MLflow credentials to a target namespace.
    
    Note: This returns a template; actual values should be read from the source and copied.
    The operator will handle reading the source secret and creating this copy.
    """
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": "mlflow-credentials",
            "namespace": target_namespace,
            "labels": build_platform_labels("mlflow", "credentials-copy"),
            "annotations": {
                "rh-agentic-operator/source-namespace": source_namespace,
            },
        },
        "type": "Opaque",
        # stringData will be filled by the operator
        "stringData": {},
    }

