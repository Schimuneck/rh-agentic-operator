"""Main Kopf operator for BaseAgent and AgenticPlatform resources."""

import os
import logging
import kopf
import kubernetes
from kubernetes import client
from kubernetes.client.rest import ApiException

from . import constants as C
from . import resources
from . import proxy
from . import platform

logger = logging.getLogger(__name__)

# Platform API constants
PLATFORM_PLURAL = "agenticplatforms"

# Operator namespace (where platform components are deployed)
OPERATOR_NAMESPACE = os.getenv("OPERATOR_NAMESPACE", "rh-agentic-system")


def get_k8s_clients():
    """Get Kubernetes API clients."""
    try:
        kubernetes.config.load_incluster_config()
    except kubernetes.config.ConfigException:
        kubernetes.config.load_kube_config()
    
    return {
        "core": client.CoreV1Api(),
        "apps": client.AppsV1Api(),
        "custom": client.CustomObjectsApi(),
        "rbac": client.RbacAuthorizationV1Api(),
    }


def apply_resource(clients: dict, resource: dict) -> None:
    """Apply a Kubernetes resource (create or update)."""
    api_version = resource.get("apiVersion", "")
    kind = resource.get("kind", "")
    name = resource["metadata"]["name"]
    namespace = resource["metadata"]["namespace"]
    
    logger.info(f"Applying {kind} {namespace}/{name}")
    
    try:
        if kind == "Deployment":
            try:
                clients["apps"].read_namespaced_deployment(name, namespace)
                clients["apps"].replace_namespaced_deployment(name, namespace, resource)
            except ApiException as e:
                if e.status == 404:
                    clients["apps"].create_namespaced_deployment(namespace, resource)
                else:
                    raise
        
        elif kind == "Service":
            try:
                existing = clients["core"].read_namespaced_service(name, namespace)
                # Preserve clusterIP
                resource["spec"]["clusterIP"] = existing.spec.cluster_ip
                clients["core"].replace_namespaced_service(name, namespace, resource)
            except ApiException as e:
                if e.status == 404:
                    clients["core"].create_namespaced_service(namespace, resource)
                else:
                    raise
        
        elif kind == "Secret":
            try:
                clients["core"].read_namespaced_secret(name, namespace)
                clients["core"].replace_namespaced_secret(name, namespace, resource)
            except ApiException as e:
                if e.status == 404:
                    clients["core"].create_namespaced_secret(namespace, resource)
                else:
                    raise
        
        elif kind == "ConfigMap":
            try:
                clients["core"].read_namespaced_config_map(name, namespace)
                clients["core"].replace_namespaced_config_map(name, namespace, resource)
            except ApiException as e:
                if e.status == 404:
                    clients["core"].create_namespaced_config_map(namespace, resource)
                else:
                    raise
        
        elif kind == "PersistentVolumeClaim":
            try:
                clients["core"].read_namespaced_persistent_volume_claim(name, namespace)
                # PVCs are immutable, skip update
                logger.info(f"PVC {namespace}/{name} already exists, skipping")
            except ApiException as e:
                if e.status == 404:
                    clients["core"].create_namespaced_persistent_volume_claim(namespace, resource)
                else:
                    raise
        
        elif kind == "ServiceAccount":
            try:
                clients["core"].read_namespaced_service_account(name, namespace)
                clients["core"].replace_namespaced_service_account(name, namespace, resource)
            except ApiException as e:
                if e.status == 404:
                    clients["core"].create_namespaced_service_account(namespace, resource)
                else:
                    raise
        
        elif kind == "Role":
            try:
                clients["rbac"].read_namespaced_role(name, namespace)
                clients["rbac"].replace_namespaced_role(name, namespace, resource)
            except ApiException as e:
                if e.status == 404:
                    clients["rbac"].create_namespaced_role(namespace, resource)
                else:
                    raise
        
        elif kind == "RoleBinding":
            try:
                clients["rbac"].read_namespaced_role_binding(name, namespace)
                clients["rbac"].replace_namespaced_role_binding(name, namespace, resource)
            except ApiException as e:
                if e.status == 404:
                    clients["rbac"].create_namespaced_role_binding(namespace, resource)
                else:
                    raise
        
        else:
            # Custom resources (ModelConfig, Agent, Route, etc.)
            group, version = api_version.split("/") if "/" in api_version else ("", api_version)
            plural = _get_plural(kind)
            
            try:
                existing = clients["custom"].get_namespaced_custom_object(
                    group, version, namespace, plural, name
                )
                # Preserve resourceVersion for update (required for routes and other resources)
                resource["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
                clients["custom"].replace_namespaced_custom_object(
                    group, version, namespace, plural, name, resource
                )
            except ApiException as e:
                if e.status == 404:
                    clients["custom"].create_namespaced_custom_object(
                        group, version, namespace, plural, resource
                    )
                else:
                    raise
    
    except ApiException as e:
        logger.error(f"Failed to apply {kind} {namespace}/{name}: {e}")
        raise


def _get_plural(kind: str) -> str:
    """Get plural form of a Kind."""
    plurals = {
        "ModelConfig": "modelconfigs",
        "Agent": "agents",
        "BaseAgent": "baseagents",
        "AgenticPlatform": "agenticplatforms",
        "Route": "routes",
        "PersistentVolumeClaim": "persistentvolumeclaims",
    }
    return plurals.get(kind, f"{kind.lower()}s")


def read_secret_value(clients: dict, namespace: str, secret_name: str, key: str) -> str:
    """Read a value from a Secret."""
    try:
        secret = clients["core"].read_namespaced_secret(secret_name, namespace)
        if secret.data and key in secret.data:
            import base64
            return base64.b64decode(secret.data[key]).decode("utf-8")
        if secret.string_data and key in secret.string_data:
            return secret.string_data[key]
    except ApiException as e:
        logger.warning(f"Could not read secret {namespace}/{secret_name}: {e}")
    return ""


def get_global_platform_config(clients: dict) -> dict:
    """Get the global AgenticPlatform config from the operator namespace.
    
    The AgenticPlatform in OPERATOR_NAMESPACE acts as the global config for all agents.
    """
    try:
        platforms = clients["custom"].list_namespaced_custom_object(
            C.API_GROUP, C.API_VERSION, OPERATOR_NAMESPACE, PLATFORM_PLURAL
        )
        if platforms.get("items"):
            p = platforms["items"][0]
            return {
                "spec": p.get("spec", {}),
                "status": p.get("status", {}),
                "namespace": OPERATOR_NAMESPACE,
                "exists": True,
            }
    except ApiException as e:
        logger.debug(f"No AgenticPlatform in operator namespace {OPERATOR_NAMESPACE}: {e}")
    return {"spec": {}, "status": {}, "namespace": OPERATOR_NAMESPACE, "exists": False}


def get_platform_for_namespace(clients: dict, namespace: str) -> dict:
    """Get the AgenticPlatform config for a namespace.
    
    First checks the operator namespace for global config, then the agent's namespace.
    """
    # First look for global platform config in operator namespace
    global_config = get_global_platform_config(clients)
    if global_config["exists"]:
        return global_config
    
    # Fall back to namespace-local platform config (for backwards compatibility)
    try:
        platforms = clients["custom"].list_namespaced_custom_object(
            C.API_GROUP, C.API_VERSION, namespace, PLATFORM_PLURAL
        )
        if platforms.get("items"):
            p = platforms["items"][0]
            return {
                "spec": p.get("spec", {}),
                "status": p.get("status", {}),
                "namespace": namespace,
                "exists": True,
            }
    except ApiException as e:
        logger.debug(f"No AgenticPlatform in {namespace}: {e}")
    return {"spec": {}, "status": {}, "namespace": namespace, "exists": False}


def merge_with_platform_defaults(spec: dict, platform_config: dict, agent_namespace: str) -> dict:
    """Merge BaseAgent spec with AgenticPlatform defaults.
    
    Args:
        spec: The BaseAgent spec
        platform_config: The platform configuration dict (includes 'namespace' key for platform location)
        agent_namespace: The namespace where the agent is deployed
    """
    if not platform_config.get("exists"):
        return spec
    
    platform_spec = platform_config["spec"]
    platform_namespace = platform_config.get("namespace", agent_namespace)
    
    # Get defaults using the platform namespace (where MLflow/Kagenti are deployed)
    defaults = platform.get_platform_defaults(platform_namespace, platform_spec)
    
    merged = dict(spec)
    
    # Inherit openai config if not specified
    if not merged.get("openai"):
        merged["openai"] = defaults.get("openai", {})
    
    # Inherit mlflow config if not specified
    if not merged.get("mlflow"):
        merged["mlflow"] = defaults.get("mlflow", {})
    
    # Inherit kagenti settings if not specified
    if "kagenti" not in merged:
        merged["kagenti"] = defaults.get("kagenti", {"enabled": True})
    
    # Store platform namespace for use by resources module
    merged["_platformNamespace"] = platform_namespace
    
    return merged


# ============================================================================
# AgenticPlatform Handlers
# ============================================================================

def apply_cluster_resource(clients: dict, resource: dict) -> None:
    """Apply a cluster-scoped resource (ClusterRole, ClusterRoleBinding)."""
    kind = resource.get("kind", "")
    name = resource["metadata"]["name"]
    
    logger.info(f"Applying cluster resource {kind} {name}")
    
    try:
        if kind == "ClusterRole":
            try:
                clients["rbac"].read_cluster_role(name)
                clients["rbac"].replace_cluster_role(name, resource)
            except ApiException as e:
                if e.status == 404:
                    clients["rbac"].create_cluster_role(resource)
                else:
                    raise
        elif kind == "ClusterRoleBinding":
            try:
                clients["rbac"].read_cluster_role_binding(name)
                clients["rbac"].replace_cluster_role_binding(name, resource)
            except ApiException as e:
                if e.status == 404:
                    clients["rbac"].create_cluster_role_binding(resource)
                else:
                    raise
        else:
            logger.warning(f"Unknown cluster resource kind: {kind}")
    except ApiException as e:
        logger.error(f"Failed to apply cluster resource {kind} {name}: {e}")
        raise


@kopf.on.create(C.API_GROUP, C.API_VERSION, PLATFORM_PLURAL)
@kopf.on.update(C.API_GROUP, C.API_VERSION, PLATFORM_PLURAL)
def reconcile_platform(spec, name, namespace, body, status, **kwargs):
    """Reconcile an AgenticPlatform resource - deploys MLflow, Kagenti, Kagent (if needed).
    
    Platform components are deployed in the same namespace as the AgenticPlatform.
    This namespace should be the operator namespace for cluster-wide access.
    """
    logger.info(f"Reconciling AgenticPlatform {namespace}/{name}")
    
    clients = get_k8s_clients()
    owner = body
    components_status = {}
    
    try:
        # 0. Label namespace for Kagenti discovery
        try:
            ns = clients["core"].read_namespace(namespace)
            labels = ns.metadata.labels or {}
            if labels.get("kagenti-enabled") != "true":
                labels["kagenti-enabled"] = "true"
                ns.metadata.labels = labels
                clients["core"].patch_namespace(namespace, ns)
                logger.info(f"Added kagenti-enabled=true label to namespace {namespace}")
        except Exception as e:
            logger.warning(f"Could not label namespace {namespace}: {e}")
        
        # 1. Deploy MLflow stack (if enabled)
        mlflow_config = spec.get("components", {}).get("mlflow", {})
        if mlflow_config.get("enabled", True):
            logger.info(f"Deploying MLflow stack in {namespace}")
            mlflow_resources = platform.build_mlflow_resources(namespace, spec, owner)
            for resource in mlflow_resources:
                apply_resource(clients, resource)
            components_status["mlflow"] = "Ready"
        else:
            components_status["mlflow"] = "Disabled"
        
        # 2. Deploy Kagenti UI (if enabled)
        kagenti_config = spec.get("components", {}).get("kagenti", {})
        if kagenti_config.get("enabled", True):
            logger.info(f"Deploying Kagenti UI in {namespace}")
            kagenti_resources = platform.build_kagenti_resources(namespace, spec, owner)
            for resource in kagenti_resources:
                apply_resource(clients, resource)
            
            # Create cluster-wide RBAC for Kagenti to list agents across namespaces
            logger.info(f"Creating cluster-wide RBAC for Kagenti UI")
            cluster_rbac = platform.build_kagenti_cluster_rbac(namespace, owner)
            for resource in cluster_rbac:
                apply_cluster_resource(clients, resource)
            
            components_status["kagenti"] = "Ready"
        else:
            components_status["kagenti"] = "Disabled"
        
        # 3. Check if Kagent needs to be deployed (only if not already installed)
        kagent_config = spec.get("components", {}).get("kagent", {})
        if kagent_config.get("enabled", False):
            if platform.check_kagent_installed(clients["custom"]):
                logger.info("Kagent is already installed cluster-wide, skipping deployment")
                components_status["kagent"] = "ExternallyManaged"
            else:
                logger.info("Kagent not found - kagent controller deployment is required")
                # Note: Full kagent controller deployment is complex and typically requires
                # cluster-admin privileges. We recommend using the official kagent installation.
                components_status["kagent"] = "NotInstalled-ManualSetupRequired"
        else:
            components_status["kagent"] = "Disabled"
        
        # 4. Store MLflow config for agents to inherit
        mlflow_auto_config = platform.get_mlflow_config_for_namespace(namespace)
        
        return {
            "phase": "Ready",
            "components": components_status,
            "mlflowConfig": mlflow_auto_config,
            "platformNamespace": namespace,
            "ready": True,
        }
    
    except Exception as e:
        logger.error(f"Failed to reconcile AgenticPlatform {namespace}/{name}: {e}", exc_info=True)
        return {
            "phase": "Failed",
            "components": components_status,
            "ready": False,
            "conditions": [{
                "type": "Ready",
                "status": "False",
                "reason": "ReconciliationFailed",
                "message": str(e)[:200],
            }],
        }


@kopf.on.delete(C.API_GROUP, C.API_VERSION, PLATFORM_PLURAL)
def delete_platform(name, namespace, **kwargs):
    """Handle AgenticPlatform deletion."""
    logger.info(f"AgenticPlatform {namespace}/{name} deleted - resources will be garbage collected")


# ============================================================================
# BaseAgent Handlers
# ============================================================================

def replicate_mlflow_credentials(clients: dict, source_ns: str, target_ns: str, owner: dict) -> None:
    """Replicate MLflow credentials from platform namespace to agent namespace.
    
    If the target namespace is different from source and credentials exist in source,
    copy them to the target namespace so agents can write to MLflow.
    """
    if source_ns == target_ns:
        return  # No need to replicate
    
    try:
        # Read source secret
        source_secret = clients["core"].read_namespaced_secret("mlflow-credentials", source_ns)
        
        # Build copy secret with same data
        copy_secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "mlflow-credentials",
                "namespace": target_ns,
                "labels": {
                    "app.kubernetes.io/managed-by": C.OPERATOR_NAME,
                    "app.kubernetes.io/component": "mlflow-credentials-copy",
                },
                "annotations": {
                    "rh-agentic-operator/source-namespace": source_ns,
                },
                "ownerReferences": [resources.build_owner_reference(owner)],
            },
            "type": "Opaque",
            "data": source_secret.data,  # Copy binary data directly
        }
        
        apply_resource(clients, copy_secret)
        logger.info(f"Replicated mlflow-credentials from {source_ns} to {target_ns}")
        
    except ApiException as e:
        if e.status == 404:
            logger.debug(f"No mlflow-credentials in {source_ns} to replicate")
        else:
            logger.warning(f"Failed to replicate mlflow-credentials: {e}")


@kopf.on.create(C.API_GROUP, C.API_VERSION, C.PLURAL)
@kopf.on.update(C.API_GROUP, C.API_VERSION, C.PLURAL)
def reconcile_baseagent(spec, name, namespace, body, status, **kwargs):
    """Reconcile a BaseAgent resource."""
    logger.info(f"Reconciling BaseAgent {namespace}/{name}")
    
    clients = get_k8s_clients()
    owner = body
    
    # Get platform config and merge defaults
    platform_config = get_platform_for_namespace(clients, namespace)
    platform_namespace = platform_config.get("namespace", namespace)
    
    if platform_config["exists"]:
        logger.info(f"Found AgenticPlatform in {platform_namespace}, inheriting defaults")
        spec = merge_with_platform_defaults(spec, platform_config, namespace)
    
    try:
        # 0. Replicate MLflow credentials from platform namespace to agent namespace
        replicate_mlflow_credentials(clients, platform_namespace, namespace, owner)
        
        # 1. Create API key secret (dummy, for kagent)
        api_key_secret = resources.build_api_key_secret(name, namespace, owner)
        apply_resource(clients, api_key_secret)
        
        # 2. Create MCP tools secret (if configured)
        mcp_tools = spec.get("mcpTools", [])
        if mcp_tools:
            # Read token values from referenced secrets
            secret_values = {}
            for tool in mcp_tools:
                secret_ref = tool.get("secretRef", {})
                if secret_ref.get("name"):
                    key = secret_ref.get("key", "token")
                    value = read_secret_value(clients, namespace, secret_ref["name"], key)
                    secret_values[secret_ref["name"]] = value
            
            mcp_secret = resources.build_mcp_secret(name, namespace, mcp_tools, owner, secret_values)
            apply_resource(clients, mcp_secret)
        
        # 3. Create A2A agents secret (if configured)
        subagents = spec.get("subagents", [])
        if subagents:
            a2a_secret = resources.build_a2a_secret(name, namespace, subagents, owner)
            apply_resource(clients, a2a_secret)
        
        # 4. Create Service
        service = resources.build_service(name, namespace, owner)
        apply_resource(clients, service)
        
        # 5. Create Deployment
        deployment = resources.build_deployment(name, namespace, spec, owner)
        apply_resource(clients, deployment)
        
        # 6. Create kagent ModelConfig
        model_config = resources.build_model_config(name, namespace, spec, owner)
        apply_resource(clients, model_config)
        
        # 7. Create kagent Agent
        kagent_agent = resources.build_kagent_agent(name, namespace, spec, owner)
        apply_resource(clients, kagent_agent)
        
        # 8. Create Kagenti resources (if enabled)
        kagenti_config = spec.get("kagenti", {})
        kagenti_enabled = kagenti_config.get("enabled", True)
        kagenti_endpoint = None
        
        if kagenti_enabled:
            # Create proxy ConfigMap
            proxy_cm = proxy.build_proxy_configmap(name, namespace, owner)
            apply_resource(clients, proxy_cm)
            
            # Create Kagenti Agent
            kagenti_agent = proxy.build_kagenti_agent(name, namespace, spec, owner)
            apply_resource(clients, kagenti_agent)
            
            kagenti_endpoint = f"http://{name}-ui.{namespace}.svc.cluster.local:{C.KAGENTI_PROXY_PORT}"
        
        # Update status
        endpoint = f"http://{name}-adapter.{namespace}.svc.cluster.local:{C.ADAPTER_PORT}"
        
        return {
            "ready": "True",
            "endpoint": endpoint,
            "kagentiEndpoint": kagenti_endpoint,
        }
    
    except Exception as e:
        logger.error(f"Failed to reconcile BaseAgent {namespace}/{name}: {e}", exc_info=True)
        return {
            "ready": "False",
            "conditions": [{
                "type": "Ready",
                "status": "False",
                "reason": "ReconciliationFailed",
                "message": str(e)[:200],
            }],
        }


@kopf.on.delete(C.API_GROUP, C.API_VERSION, C.PLURAL)
def delete_baseagent(name, namespace, **kwargs):
    """Handle BaseAgent deletion.
    
    Resources are cleaned up automatically via ownerReferences.
    """
    logger.info(f"BaseAgent {namespace}/{name} deleted - resources will be garbage collected")


@kopf.timer(C.API_GROUP, C.API_VERSION, C.PLURAL, interval=60.0)
def check_health(spec, name, namespace, status, **kwargs):
    """Periodic health check for BaseAgent."""
    clients = get_k8s_clients()
    
    try:
        # Check if deployment is ready
        deployment = clients["apps"].read_namespaced_deployment(f"{name}-adapter", namespace)
        ready_replicas = deployment.status.ready_replicas or 0
        desired_replicas = deployment.spec.replicas or 1
        
        if ready_replicas >= desired_replicas:
            return {"ready": "True"}
        else:
            return {
                "ready": "False",
                "conditions": [{
                    "type": "Ready",
                    "status": "False",
                    "reason": "DeploymentNotReady",
                    "message": f"Ready replicas: {ready_replicas}/{desired_replicas}",
                }],
            }
    except ApiException as e:
        if e.status == 404:
            return {
                "ready": "False",
                "conditions": [{
                    "type": "Ready",
                    "status": "False",
                    "reason": "DeploymentNotFound",
                    "message": "Adapter deployment not found",
                }],
            }
        raise


def main():
    """Entry point for the operator."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Kopf takes over from here
    kopf.run()


if __name__ == "__main__":
    main()

