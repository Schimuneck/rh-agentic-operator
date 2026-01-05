# rh-agentic-operator Makefile

NAMESPACE ?= mschimun
IMAGE_REGISTRY ?= image-registry.openshift-image-registry.svc:5000
OPERATOR_IMAGE = $(IMAGE_REGISTRY)/$(NAMESPACE)/rh-agentic-operator:latest
BASEAGENT_IMAGE = $(IMAGE_REGISTRY)/$(NAMESPACE)/base-agent:latest

.PHONY: all build push deploy undeploy install-crd uninstall-crd build-baseagent

all: build push deploy

# ===========================================
# Build
# ===========================================

build:
	@echo "Building operator image..."
	oc start-build rh-agentic-operator --from-dir=. -n $(NAMESPACE) --follow || \
		(oc apply -f config/buildconfig-operator.yaml -n $(NAMESPACE) && \
		 oc start-build rh-agentic-operator --from-dir=. -n $(NAMESPACE) --follow)

build-baseagent:
	@echo "Building base-agent image..."
	oc start-build base-agent --from-dir=base-agent -n $(NAMESPACE) --follow || \
		(oc apply -f base-agent/k8s/buildconfig.yaml -n $(NAMESPACE) && \
		 oc start-build base-agent --from-dir=base-agent -n $(NAMESPACE) --follow)

build-all: build-baseagent build

# ===========================================
# Deploy
# ===========================================

install-crd:
	@echo "Installing BaseAgent CRD..."
	oc apply -f config/crd.yaml

uninstall-crd:
	@echo "Removing BaseAgent CRD..."
	oc delete -f config/crd.yaml --ignore-not-found

deploy: install-crd
	@echo "Deploying operator to namespace $(NAMESPACE)..."
	@# Replace namespace placeholder
	@sed 's/NAMESPACE_PLACEHOLDER/$(NAMESPACE)/g' config/rbac.yaml | oc apply -f -
	@sed 's/NAMESPACE_PLACEHOLDER/$(NAMESPACE)/g' config/deployment.yaml | oc apply -f -
	@echo "Waiting for operator to be ready..."
	oc rollout status deployment/rh-agentic-operator -n $(NAMESPACE) --timeout=120s

undeploy:
	@echo "Removing operator from namespace $(NAMESPACE)..."
	@sed 's/NAMESPACE_PLACEHOLDER/$(NAMESPACE)/g' config/deployment.yaml | oc delete -f - --ignore-not-found
	@sed 's/NAMESPACE_PLACEHOLDER/$(NAMESPACE)/g' config/rbac.yaml | oc delete -f - --ignore-not-found

# ===========================================
# Development
# ===========================================

run-local:
	@echo "Running operator locally..."
	PYTHONPATH=src kopf run --standalone -m rh_agentic_operator.operator

logs:
	oc logs -f deployment/rh-agentic-operator -n $(NAMESPACE)

status:
	@echo "=== Operator ==="
	oc get deployment rh-agentic-operator -n $(NAMESPACE)
	@echo ""
	@echo "=== BaseAgents ==="
	oc get baseagents -n $(NAMESPACE) || echo "No BaseAgents found"

# ===========================================
# Examples
# ===========================================

example-rag:
	@echo "Deploying example RAG agent..."
	oc apply -f examples/simple-rag-agent.yaml -n $(NAMESPACE)

example-mcp:
	@echo "Deploying example MCP agent..."
	oc apply -f examples/mcp-agent.yaml -n $(NAMESPACE)

example-orchestrator:
	@echo "Deploying example orchestrator..."
	oc apply -f examples/orchestrator-agent.yaml -n $(NAMESPACE)

# ===========================================
# Clean
# ===========================================

clean:
	@echo "Removing all BaseAgents in namespace $(NAMESPACE)..."
	oc delete baseagents --all -n $(NAMESPACE)

clean-all: clean undeploy uninstall-crd

# ===========================================
# Help
# ===========================================

help:
	@echo "rh-agentic-operator Makefile"
	@echo ""
	@echo "Usage: make [target] NAMESPACE=<namespace>"
	@echo ""
	@echo "Targets:"
	@echo "  build           - Build operator image"
	@echo "  build-baseagent - Build base-agent image"
	@echo "  build-all       - Build both images"
	@echo "  deploy          - Deploy operator (includes CRD)"
	@echo "  undeploy        - Remove operator"
	@echo "  install-crd     - Install BaseAgent CRD only"
	@echo "  uninstall-crd   - Remove BaseAgent CRD"
	@echo "  run-local       - Run operator locally for development"
	@echo "  logs            - Show operator logs"
	@echo "  status          - Show operator and BaseAgent status"
	@echo "  example-rag     - Deploy simple RAG agent example"
	@echo "  example-mcp     - Deploy MCP agent example"
	@echo "  example-orchestrator - Deploy orchestrator example"
	@echo "  clean           - Remove all BaseAgents"
	@echo "  clean-all       - Remove everything"

