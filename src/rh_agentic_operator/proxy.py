"""Kagenti proxy configuration builders."""

from typing import Any, Dict

from . import constants as C
from .resources import build_labels, build_owner_reference


NGINX_CONF_TEMPLATE = """
worker_processes  1;

pid /tmp/nginx.pid;
error_log /tmp/nginx-error.log info;

events {{ worker_connections  1024; }}

http {{
  include       /etc/nginx/mime.types;
  default_type  application/octet-stream;

  access_log /tmp/nginx-access.log;

  sendfile on;
  tcp_nopush on;
  tcp_nodelay on;
  keepalive_timeout  65;

  # Temp paths for non-root
  client_body_temp_path /tmp/nginx/client_body;
  proxy_temp_path       /tmp/nginx/proxy_temp;
  fastcgi_temp_path     /tmp/nginx/fastcgi_temp;
  uwsgi_temp_path       /tmp/nginx/uwsgi_temp;
  scgi_temp_path        /tmp/nginx/scgi_temp;

  # SSE/streaming friendliness
  proxy_buffering off;
  proxy_cache off;

  server {{
    listen {proxy_port};

    location / {{
      proxy_http_version 1.1;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto $scheme;

      # Forward to the adapter service
      proxy_pass http://{service_name}.{namespace}.svc.cluster.local:{adapter_port};
    }}
  }}
}}
"""


def build_proxy_configmap(name: str, namespace: str, owner: Dict[str, Any]) -> Dict[str, Any]:
    """Build ConfigMap containing nginx.conf for Kagenti proxy.
    
    The proxy forwards to the kagent-managed service (which handles A2A protocol),
    NOT directly to our adapter. Kagent creates a service named after the Agent
    that provides A2A → OpenAI translation.
    """
    nginx_conf = NGINX_CONF_TEMPLATE.format(
        proxy_port=C.KAGENTI_PROXY_PORT,
        # Forward to kagent's service (handles A2A), not our adapter
        service_name=name,  # kagent creates service with agent name
        namespace=namespace,
        adapter_port=C.ADAPTER_PORT,  # kagent service also uses 8080
    )
    
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": f"{name}-proxy-config",
            "namespace": namespace,
            "labels": build_labels(name, "proxy"),
            "ownerReferences": [build_owner_reference(owner)],
        },
        "data": {
            "nginx.conf": nginx_conf,
        },
    }


def build_kagenti_agent(name: str, namespace: str, spec: Dict[str, Any], owner: Dict[str, Any]) -> Dict[str, Any]:
    """Build Kagenti Agent CR with embedded nginx proxy.
    
    This creates a kagenti.dev Agent that Kagenti UI can discover.
    The agent runs an nginx proxy that forwards requests to the base-agent adapter.
    """
    agent_card = spec.get("agentCard", {})
    
    return {
        "apiVersion": "agent.kagenti.dev/v1alpha1",
        "kind": "Agent",
        "metadata": {
            "name": f"{name}-ui",
            "namespace": namespace,
            "labels": {
                **build_labels(name, "kagenti"),
                "kagenti.io/type": "agent",
                "kagenti.io/protocol": "a2a",
                "kagenti.io/framework": "kagent",
            },
            "ownerReferences": [build_owner_reference(owner)],
        },
        "spec": {
            "description": agent_card.get("description", f"Kagenti wrapper for {name}"),
            "replicas": 1,
            "imageSource": {
                "image": C.KAGENTI_PROXY_IMAGE,
            },
            "servicePorts": [{
                "name": "http",
                "port": C.KAGENTI_PROXY_PORT,
                "targetPort": C.KAGENTI_PROXY_PORT,
                "protocol": "TCP",
            }],
            "podTemplateSpec": {
                "metadata": {
                    "labels": {
                        # Kagenti looks for pods with app.kubernetes.io/name matching agent name
                        "app.kubernetes.io/name": f"{name}-ui",
                        "app.kubernetes.io/component": "agent",
                        "app.kubernetes.io/instance": name,
                        "app.kubernetes.io/managed-by": "rh-agentic-operator",
                        "kagenti.io/type": "agent",
                        "kagenti.io/protocol": "a2a",
                        "kagenti.io/framework": "kagent",
                    },
                },
                "spec": {
                    "containers": [{
                        "name": "proxy",
                        "image": C.KAGENTI_PROXY_IMAGE,
                        "command": [
                            "sh", "-c",
                            "mkdir -p /tmp/nginx/client_body /tmp/nginx/proxy_temp /tmp/nginx/fastcgi_temp /tmp/nginx/uwsgi_temp /tmp/nginx/scgi_temp && nginx -g 'daemon off;'"
                        ],
                        "ports": [{
                            "containerPort": C.KAGENTI_PROXY_PORT,
                            "protocol": "TCP",
                        }],
                        "volumeMounts": [
                            {
                                "name": "nginx-config",
                                "mountPath": "/etc/nginx/nginx.conf",
                                "subPath": "nginx.conf",
                            },
                            {
                                "name": "cache",
                                "mountPath": "/tmp",
                            },
                        ],
                        "readinessProbe": {
                            "tcpSocket": {"port": C.KAGENTI_PROXY_PORT},
                            "initialDelaySeconds": 3,
                            "periodSeconds": 10,
                        },
                        "livenessProbe": {
                            "tcpSocket": {"port": C.KAGENTI_PROXY_PORT},
                            "initialDelaySeconds": 10,
                            "periodSeconds": 30,
                        },
                        "resources": {
                            "requests": {"cpu": "10m", "memory": "32Mi"},
                            "limits": {"cpu": "100m", "memory": "128Mi"},
                        },
                    }],
                    "volumes": [
                        {
                            "name": "nginx-config",
                            "configMap": {
                                "name": f"{name}-proxy-config",
                            },
                        },
                        {
                            "name": "cache",
                            "emptyDir": {},
                        },
                    ],
                },
            },
        },
    }

