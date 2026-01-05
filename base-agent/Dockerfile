FROM registry.access.redhat.com/ubi9/python-311:latest

USER 0
RUN dnf install -y --nodocs git && dnf clean all

USER 1001

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY agent.py server.py ./

# Environment defaults
ENV HOME=/tmp
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
