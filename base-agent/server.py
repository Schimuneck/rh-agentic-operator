"""
OpenAI-compatible API server for kagent.

This server:
- Provides /v1/chat/completions endpoint (OpenAI format)
- Uses BaseAgent with native RAG/MCP via Llama Stack
- Supports A2A subagent orchestration
- Logs all interactions to MLflow (when configured)
"""
import os
import json
import logging
from typing import Optional
from uuid import uuid4
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import asyncio

# MLflow tracking is optional
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
MLFLOW_ENABLED = bool(MLFLOW_TRACKING_URI)

if MLFLOW_ENABLED:
    import mlflow
    from mlflow.types.responses import ResponsesAgentRequest
else:
    # Stub for when MLflow is disabled
    mlflow = None
    class ResponsesAgentRequest:
        def __init__(self, input=None):
            self.input = input or []

from agent import BaseAgent

logger = logging.getLogger(__name__)

# Global agent
agent: Optional[BaseAgent] = None

# MLflow config
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "base-agent")


def init_mlflow():
    """Initialize MLflow tracking."""
    if MLFLOW_ENABLED and mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        logger.info(f"MLflow tracking enabled: {MLFLOW_TRACKING_URI}, experiment: {MLFLOW_EXPERIMENT}")
    else:
        logger.info("MLflow tracking disabled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    init_mlflow()
    logger.info("Initializing BaseAgent...")
    agent = BaseAgent()
    logger.info("Agent ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="base-agent",
    description="MLflow ResponsesAgent orchestrator with RAG/MCP/A2A for kagent",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/healthz")
def healthz():
    """Health check."""
    return {
        "ok": True,
        "agent_ready": agent is not None,
        "mlflow_enabled": MLFLOW_ENABLED,
        "vector_stores": agent.vector_store_ids if agent else [],
        "mcp_tools": len(agent.mcp_tools) if agent else 0,
        "a2a_agents": len(agent.a2a_agents) if agent else 0,
    }


# A2A Agent Card - required for Kagenti to discover and interact with the agent
AGENT_NAME = os.getenv("AGENT_NAME", "base-agent")
AGENT_DESCRIPTION = os.getenv("AGENT_DESCRIPTION", "An AI agent powered by Llama Stack")
AGENT_VERSION = os.getenv("AGENT_VERSION", "1.0.0")


@app.get("/.well-known/agent.json")
def agent_card():
    """A2A Agent Card - describes this agent's capabilities for discovery."""
    skills = []
    
    # Add chat skill
    skills.append({
        "id": "chat",
        "name": "Chat",
        "description": "General conversation and question answering",
        "tags": ["chat", "qa"],
        "examples": ["Hello, how can you help me?", "What can you do?"]
    })
    
    # Add RAG skill if configured
    if agent and agent.vector_store_ids:
        skills.append({
            "id": "rag",
            "name": "Knowledge Search",
            "description": "Search and retrieve information from vector stores",
            "tags": ["rag", "search", "knowledge"],
            "examples": ["Search for information about...", "What do you know about..."]
        })
    
    # Add MCP tools skill if configured
    if agent and agent.mcp_tools:
        tool_labels = [t.get("server_label", "unknown") for t in agent.mcp_tools]
        skills.append({
            "id": "mcp",
            "name": "External Tools",
            "description": f"Access external services via MCP: {', '.join(tool_labels)}",
            "tags": ["mcp", "tools", "external"],
            "examples": ["Look up customer data", "Query external systems"]
        })
    
    # Add orchestration skill if subagents configured
    if agent and agent.a2a_agents:
        skills.append({
            "id": "orchestrate",
            "name": "Orchestration",
            "description": f"Coordinates with {len(agent.a2a_agents)} specialized agents",
            "tags": ["orchestration", "multi-agent"],
            "examples": ["Help me with a complex task", "I need multiple experts"]
        })
    
    return {
        "name": AGENT_NAME,
        "description": AGENT_DESCRIPTION,
        "version": AGENT_VERSION,
        "url": f"http://{AGENT_NAME}-adapter:8080",
        "protocol": "a2a",
        "protocolVersion": "1.0",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": False
        },
        "skills": skills,
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"]
    }


def extract_output_text(response) -> str:
    """Extract text from agent response output items."""
    output_text = ""
    for item in response.output:
        if hasattr(item, "content"):
            content = item.content
            if isinstance(content, str):
                output_text += content
            elif hasattr(content, "text"):
                output_text += content.text or ""
            elif isinstance(content, list):
                for c in content:
                    if hasattr(c, "text"):
                        output_text += c.text or ""
                    elif isinstance(c, dict) and "text" in c:
                        output_text += c.get("text", "")
        elif hasattr(item, "text"):
            output_text += item.text or ""
        elif isinstance(item, dict) and "text" in item:
            output_text += item["text"]
    return output_text


async def generate_sse_stream(agent_obj, agent_request, req_id: str):
    """Generate Server-Sent Events for streaming response."""
    try:
        # Use predict_stream if available, otherwise fall back to predict
        if hasattr(agent_obj, 'predict_stream'):
            full_text = ""
            for event in agent_obj.predict_stream(agent_request):
                if event.type == "response.output_item.done":
                    # Extract text from the event item
                    item = event.item
                    chunk_text = ""
                    if hasattr(item, "text"):
                        chunk_text = item.text or ""
                    elif hasattr(item, "content"):
                        content = item.content
                        if isinstance(content, str):
                            chunk_text = content
                        elif hasattr(content, "text"):
                            chunk_text = content.text or ""
                        elif isinstance(content, list) and content:
                            for c in content:
                                if hasattr(c, "text"):
                                    chunk_text += c.text or ""
                                elif isinstance(c, dict) and "text" in c:
                                    chunk_text += c.get("text", "")
                    
                    full_text = chunk_text
                    
                    # Send the chunk as SSE
                    chunk_data = {
                        "id": f"chatcmpl-{req_id[:12]}",
                        "object": "chat.completion.chunk",
                        "model": agent_obj.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
            
            # Send the final message with finish_reason
            final_chunk = {
                "id": f"chatcmpl-{req_id[:12]}",
                "object": "chat.completion.chunk",
                "model": agent_obj.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        else:
            # Fallback: get full response and stream it
            response = agent_obj.predict(agent_request)
            output_text = extract_output_text(response)
            
            # Send as a single chunk
            chunk_data = {
                "id": f"chatcmpl-{req_id[:12]}",
                "object": "chat.completion.chunk",
                "model": agent_obj.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": output_text},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
            
            final_chunk = {
                "id": f"chatcmpl-{req_id[:12]}",
                "object": "chat.completion.chunk",
                "model": agent_obj.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        error_data = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint with streaming support."""
    global agent
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")
    
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    
    if not messages:
        raise HTTPException(status_code=400, detail="Missing messages")
    
    req_id = str(uuid4())
    
    # Extract session/user IDs from headers or body for MLflow trace association
    # Headers: X-User-Id, X-Session-Id
    # Body: user_id, session_id (in metadata or root)
    user_id = (
        request.headers.get("X-User-Id") or 
        body.get("user_id") or 
        body.get("metadata", {}).get("user_id") or
        "anonymous"
    )
    session_id = (
        request.headers.get("X-Session-Id") or 
        body.get("session_id") or 
        body.get("metadata", {}).get("session_id")
    )
    
    # Build agent request with metadata
    agent_request = ResponsesAgentRequest(
        input=[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
    )
    # Attach metadata for session tracking
    agent_request.metadata = {"user_id": user_id, "session_id": session_id}
    
    # Handle streaming response
    if stream:
        logger.info(f"Streaming request {req_id[:8]}")
        return StreamingResponse(
            generate_sse_stream(agent, agent_request, req_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
    
    # Non-streaming response
    run = None
    
    if MLFLOW_ENABLED and mlflow:
        run = mlflow.start_run(run_name=f"chat-{req_id[:8]}")
        mlflow.set_tag("component", "base-agent")
        mlflow.set_tag("request_id", req_id)
        mlflow.set_tag("model", agent.model)
        
        # Session tracking tags (using trace.* prefix for MLflow trace association)
        mlflow.set_tag("trace.user_id", user_id)
        if session_id:
            mlflow.set_tag("trace.session_id", session_id)
        mlflow.set_tag("trace.agent_name", AGENT_NAME)
        
        if agent.vector_store_ids:
            mlflow.set_tag("vector_stores", ",".join(agent.vector_store_ids))
        if agent.mcp_tools:
            mlflow.set_tag("mcp_tools", ",".join(t.get("server_label", "?") for t in agent.mcp_tools))
        if agent.a2a_agents:
            mlflow.set_tag("a2a_agents", ",".join(a["name"] for a in agent.a2a_agents[:10]))
    
    try:
        # Get response
        response = agent.predict(agent_request)
        
        # Extract text
        output_text = extract_output_text(response)
        
        # Log to MLflow
        if run and mlflow:
            user_msg = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
            mlflow.log_text(user_msg[:4000], "prompt.txt")
            mlflow.log_text(output_text[:20000], "response.txt")
            
            # Log routing if orchestration was used
            if agent.last_routing:
                try:
                    mlflow.log_dict(agent.last_routing, "routing.json")
                    selected = agent.last_routing.get("selected", [])
                    if selected:
                        mlflow.set_tag("selected_agents", ",".join(s["name"] for s in selected[:10]))
                except Exception as e:
                    logger.warning(f"Failed to log routing: {e}")
            
            if agent.last_subagent_results:
                try:
                    # Log full results as JSON
                    mlflow.log_dict({"qa_pairs": agent.last_subagent_results}, "subagent_results.json")
                    # Also log a readable summary
                    summary_lines = []
                    for r in agent.last_subagent_results:
                        summary_lines.append(f"## {r.get('name')} ({r.get('elapsed_ms', 0)}ms)")
                        summary_lines.append(f"**Question:** {r.get('question', 'N/A')}")
                        summary_lines.append(f"**Answer:** {r.get('answer', 'N/A')[:2000]}")
                        summary_lines.append("")
                    mlflow.log_text("\n".join(summary_lines), "subagent_qa.md")
                except Exception as e:
                    logger.warning(f"Failed to log subagent results: {e}")
        
        result = {
            "id": f"chatcmpl-{req_id[:12]}",
            "object": "chat.completion",
            "model": agent.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        if run and mlflow:
            mlflow.set_tag("error", str(e)[:250])
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if run and mlflow:
            mlflow.end_run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
