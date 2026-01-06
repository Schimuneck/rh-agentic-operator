FROM registry.access.redhat.com/ubi9/python-311:latest

USER 0

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY setup.py .
COPY src/ src/

# Install the operator package
RUN pip install --no-cache-dir -e .

USER 1001

# Kopf runs on port 8080 by default for health checks
EXPOSE 8080

# Run the operator
CMD ["kopf", "run", "--standalone", "--all-namespaces", "-m", "rh_agentic_operator.operator"]



