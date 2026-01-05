"""
Base-agent: MLflow ResponsesAgent with A2A orchestration.

Features:
- OpenAI client pointing to Llama Stack /v1/responses
- Native RAG via file_search tool
- Native MCP tools
- A2A subagent orchestration with intelligent question routing
- Conversation continuity via previous_response_id
- MLflow ResponsesAgent for automatic tracing

Orchestration Flow:
1. Fetch agent cards from all configured subagents
2. Group subagents by batch
3. For each batch, ask LLM to select relevant agents AND craft specific questions
4. Call each selected agent with its tailored question
5. Format responses as Q&A pairs for final synthesis

Configuration (environment variables):
- LLAMASTACK_URL: Llama Stack base URL
- LLAMASTACK_MODEL: Model ID
- VECTOR_STORE_IDS: Comma-separated or JSON array of vector store IDs
- MAX_RESULTS: Max results per vector store search (default: 10)
- MCP_TOOLS: JSON array of MCP server configs
- A2A_AGENTS_JSON: JSON array of A2A subagents to consult
- A2A_SELECTION_DEFAULT_BATCH: Default batch for agents without batch field
- A2A_MAX_SELECTED: Max subagents to call per request (0 = no limit)
- A2A_CALL_TIMEOUT_S: HTTP timeout for A2A calls
"""
import os
import json
import logging
import time
import concurrent.futures
from typing import Generator, List, Dict, Any, Optional, Tuple
from uuid import uuid4

import httpx
from openai import OpenAI

# MLflow tracking is optional
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
MLFLOW_ENABLED = bool(MLFLOW_TRACKING_URI)

if MLFLOW_ENABLED:
    from mlflow.pyfunc import ResponsesAgent
    from mlflow.types.responses import (
        ResponsesAgentRequest,
        ResponsesAgentResponse,
        ResponsesAgentStreamEvent,
    )
    from mlflow.models import set_model
    BaseClass = ResponsesAgent
else:
    # Stubs for when MLflow is disabled
    class ResponsesAgentRequest:
        def __init__(self, input=None):
            self.input = input or []
    
    class ResponsesAgentResponse:
        def __init__(self, output=None, **kwargs):
            self.output = output or []
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ResponsesAgentStreamEvent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class OutputItem:
        def __init__(self, text=None, id=None):
            self.text = text
            self.id = id
    
    def set_model(model):
        pass
    
    class BaseClassStub:
        """Stub base class when MLflow is disabled."""
        def create_text_output_item(self, text: str, id: str = None) -> OutputItem:
            return OutputItem(text=text, id=id or str(uuid4()))
    
    BaseClass = BaseClassStub

logger = logging.getLogger(__name__)


# =============================================================================
# A2A Routing System Instruction (cannot be overridden)
# =============================================================================

A2A_ROUTING_INSTRUCTION = """You are an intelligent orchestrator that routes questions to specialized agents.

CRITICAL RULES:
1. You MUST respond with ONLY valid JSON - no markdown, no explanations, no extra text
2. Analyze the user's question and available agents carefully
3. Select only agents that are genuinely relevant to answering the question
4. For each selected agent, craft a SPECIFIC question tailored to that agent's expertise
5. The question should extract exactly the information needed from that agent
6. Do NOT just repeat the user's question - make it specific to the agent's capabilities
7. If no agents are relevant, return an empty selection

OUTPUT FORMAT (strict JSON only):
{
  "selected": [
    {
      "name": "agent-name-exactly-as-provided",
      "question": "specific question tailored for this agent",
      "reason": "brief reason why this agent was selected"
    }
  ]
}"""

A2A_SYNTHESIS_INSTRUCTION = """You are an intelligent assistant synthesizing information from multiple sources.

CRITICAL RULES:
1. Answer the user's original question comprehensively
2. Use the information from the consulted agents when relevant
3. You may also use your tools (RAG, MCP) if additional information is needed
4. Cite which agent provided which information when appropriate
5. Be clear, accurate, and well-organized in your response
6. If agents provided conflicting information, acknowledge and explain the differences
7. Follow any additional instructions provided by the system"""


# =============================================================================
# Environment Parsing
# =============================================================================

def parse_list_env(env_var: str, fallback_env: str = None, default: List[str] = None) -> List[str]:
    """Parse environment variable as list (JSON array or comma-separated)."""
    value = os.getenv(env_var, "").strip()
    if not value and fallback_env:
        value = os.getenv(fallback_env, "").strip()
    if not value:
        return default or []
    if value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_mcp_tools_env() -> List[Dict[str, Any]]:
    """Parse MCP_TOOLS environment variable."""
    value = os.getenv("MCP_TOOLS", "").strip()
    if not value:
        return []
    try:
        tools = json.loads(value)
        if not isinstance(tools, list):
            return []
        valid = []
        for t in tools:
            if isinstance(t, dict) and t.get("server_url") and t.get("server_label"):
                valid.append(t)
        return valid
    except json.JSONDecodeError:
        return []


def parse_a2a_agents_env() -> List[Dict[str, Any]]:
    """Parse A2A_AGENTS_JSON environment variable."""
    value = os.getenv("A2A_AGENTS_JSON", "").strip()
    if not value:
        return []
    try:
        agents = json.loads(value)
        if not isinstance(agents, list):
            return []
        valid = []
        for a in agents:
            if isinstance(a, dict) and a.get("name") and a.get("url"):
                valid.append(a)
        return valid
    except json.JSONDecodeError:
        return []


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(BaseClass):
    """
    Base-agent: orchestrator with RAG, MCP, and A2A subagent support.
    
    When A2A_AGENTS_JSON is configured:
    1. Fetches agent cards from all subagents
    2. Groups subagents by batch
    3. For each batch, asks LLM to:
       - Select which agents are relevant
       - Craft a specific question for each selected agent
    4. Calls selected agents with their tailored questions
    5. Synthesizes final answer using:
       - User's original question
       - Q&A pairs from subagents
       - RAG context
       - MCP tools
    
    When no subagents configured, behaves like a standard RAG+MCP agent.
    """
    
    def __init__(self):
        super().__init__()
        
        # Agent instruction (system prompt)
        self.instruction = os.getenv("AGENT_INSTRUCTION", "").strip()
        
        # Llama Stack config
        # Support both OPENAI_BASE_URL (new) and LLAMASTACK_URL (legacy)
        base = os.getenv("OPENAI_BASE_URL", os.getenv("LLAMASTACK_URL", "http://llama-stack-service:8321"))
        self.base_url = base.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        # Support both OPENAI_MODEL (new) and LLAMASTACK_MODEL (legacy)
        self.model = os.getenv("OPENAI_MODEL", os.getenv("LLAMASTACK_MODEL", "vllm-inference-1/qwen3-14b-awq"))
        self.max_results = int(os.getenv("MAX_RESULTS", "10"))
        
        # RAG + MCP
        self.vector_store_ids = parse_list_env("VECTOR_STORE_IDS", fallback_env="VECTOR_STORE_ID")
        self.mcp_tools = parse_mcp_tools_env()
        
        # A2A subagents
        self.a2a_agents = parse_a2a_agents_env()
        self.a2a_default_batch = os.getenv("A2A_SELECTION_DEFAULT_BATCH", "default")
        self.a2a_call_timeout = float(os.getenv("A2A_CALL_TIMEOUT_S", "60"))
        self.a2a_max_selected = int(os.getenv("A2A_MAX_SELECTED", "4"))
        
        # OpenAI client for Llama Stack
        api_key = os.getenv("OPENAI_API_KEY", "not-needed")
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)
        
        # Conversation state
        self._last_response_id: Optional[str] = None
        
        # Exposed for MLflow logging
        self.last_routing: Optional[Dict[str, Any]] = None
        self.last_subagent_results: Optional[List[Dict[str, Any]]] = None
        
        self._log_config()
    
    def _log_config(self):
        logger.info("BaseAgent initialized")
        logger.info(f"  Instruction: {self.instruction[:100]}..." if self.instruction else "  Instruction: (none)")
        logger.info(f"  Llama Stack: {self.base_url}")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Vector stores: {self.vector_store_ids or '(none)'}")
        logger.info(f"  MCP tools: {len(self.mcp_tools)}")
        logger.info(f"  A2A agents: {len(self.a2a_agents)}")
        for a in self.a2a_agents[:10]:
            logger.info(f"    - {a['name']} (batch={a.get('batch', self.a2a_default_batch)})")
    
    # =========================================================================
    # Tool Building
    # =========================================================================
    
    def _build_tools(self) -> List[Dict[str, Any]]:
        """Build tools list for Llama Stack responses API."""
        tools = []
        
        # RAG via file_search
        if self.vector_store_ids:
            tools.append({
                "type": "file_search",
                "vector_store_ids": self.vector_store_ids,
                "max_num_results": self.max_results,
            })
        
        # MCP tools
        for mcp in self.mcp_tools:
            tool = {
                "type": "mcp",
                "server_url": mcp["server_url"],
                "server_label": mcp["server_label"],
            }
            if mcp.get("authorization"):
                tool["authorization"] = mcp["authorization"]
            if mcp.get("headers"):
                tool["headers"] = mcp["headers"]
            if mcp.get("allowed_tools"):
                tool["allowed_tools"] = mcp["allowed_tools"]
            tools.append(tool)
        
        return tools
    
    # =========================================================================
    # Llama Stack API Helpers
    # =========================================================================
    
    def _build_input_with_instruction(self, input_text: str, include_instruction: bool = True) -> str:
        """Build input with optional instruction prefix."""
        if include_instruction and self.instruction:
            return f"""SYSTEM INSTRUCTION:
{self.instruction}

USER REQUEST:
{input_text}"""
        return input_text
    
    def _responses_create(
        self, input_text: str, tools: Optional[List] = None, use_previous: bool = False,
        include_instruction: bool = True
    ) -> Tuple[Any, str]:
        """Call Llama Stack /v1/responses and return (response, output_text)."""
        full_input = self._build_input_with_instruction(input_text, include_instruction)
        kwargs = {"model": self.model, "input": full_input}
        if tools:
            kwargs["tools"] = tools
        if use_previous and self._last_response_id:
            kwargs["previous_response_id"] = self._last_response_id
        
        response = self.client.responses.create(**kwargs)
        text = self._extract_text(response)
        return response, text
    
    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract output text from responses API result."""
        parts = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        text = getattr(c, "text", "") or ""
                        # Strip thinking tags
                        if "</think>" in text:
                            text = text.split("</think>")[-1].strip()
                        parts.append(text)
        return "".join(parts).strip()
    
    # =========================================================================
    # A2A Orchestration
    # =========================================================================
    
    def _fetch_agent_card(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch A2A agent card."""
        try:
            with httpx.Client(timeout=min(self.a2a_call_timeout, 15)) as client:
                r = client.get(url.rstrip("/") + "/.well-known/agent.json")
                r.raise_for_status()
                return r.json() if isinstance(r.json(), dict) else None
        except Exception as e:
            logger.warning(f"Failed to fetch agent card from {url}: {e}")
            return None
    
    def _fetch_all_agent_cards(self, agents: List[Dict]) -> List[Dict]:
        """Fetch agent cards in parallel."""
        def fetch_one(agent):
            card = self._fetch_agent_card(agent["url"])
            return {**agent, "card": card}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(fetch_one, agents))
    
    def _select_agents_with_questions(
        self, batch_name: str, user_text: str, candidates: List[Dict]
    ) -> List[Dict]:
        """
        Ask LLM to select relevant agents AND craft specific questions for each.
        
        Returns list of:
        {
            "name": "agent-name",
            "question": "specific question tailored for this agent",
            "reason": "why this agent was selected"
        }
        """
        if not candidates:
            return []
        
        # Build compact candidate descriptions
        descs = []
        for c in candidates:
            card = c.get("card") or {}
            skills = card.get("skills", [])[:5]
            skill_info = []
            for s in skills:
                if isinstance(s, dict):
                    skill_info.append({
                        "id": s.get("id", ""),
                        "name": s.get("name", ""),
                        "description": s.get("description", "")[:200],
                    })
            descs.append({
                "name": c["name"],
                "description": card.get("description", "")[:300],
                "skills": skill_info,
            })
        
        # Use fixed A2A routing instruction (cannot be overridden by agent instruction)
        prompt = f"""{A2A_ROUTING_INSTRUCTION}

---

USER'S QUESTION:
{user_text}

AVAILABLE AGENTS IN BATCH "{batch_name}":
{json.dumps(descs, indent=2)}

RESPOND NOW WITH JSON ONLY:"""

        try:
            # Internal routing call - don't include agent instruction
            _, text = self._responses_create(prompt, tools=None, use_previous=False, include_instruction=False)
            parsed = self._parse_json(text)
            selected = parsed.get("selected", [])
            if not isinstance(selected, list):
                return []
            
            valid_names = {c["name"] for c in candidates}
            result = []
            for s in selected:
                if not isinstance(s, dict):
                    continue
                name = s.get("name")
                if name not in valid_names:
                    continue
                question = s.get("question", "").strip()
                if not question:
                    # Fallback to user's original question if LLM didn't provide one
                    question = user_text
                result.append({
                    "name": name,
                    "question": question[:2000],  # Limit question length
                    "reason": s.get("reason", "")[:200],
                })
            return result
        except Exception as e:
            logger.warning(f"Selection failed for batch {batch_name}: {e}")
            return []
    
    @staticmethod
    def _parse_json(text: str) -> Dict:
        """Parse JSON from model output (robust to surrounding text)."""
        text = (text or "").strip()
        try:
            return json.loads(text) if isinstance(json.loads(text), dict) else {}
        except:
            pass
        # Find first {...} block
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
        return {}
    
    def _call_a2a_agent(self, url: str, question: str) -> str:
        """Call an A2A agent via JSON-RPC message/send with a specific question."""
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": f"m_{uuid4().hex[:8]}",
                    "role": "user",
                    "parts": [{"kind": "text", "text": question}],
                }
            },
        }
        try:
            with httpx.Client(timeout=self.a2a_call_timeout) as client:
                r = client.post(url.rstrip("/") + "/", json=payload)
                r.raise_for_status()
                data = r.json()
            
            # Extract text from response
            texts = self._extract_texts_recursive(data)
            return "\n".join(texts).strip()[:15000] or json.dumps(data)[:15000]
        except Exception as e:
            return f"Error calling {url}: {e}"
    
    @staticmethod
    def _extract_texts_recursive(obj: Any) -> List[str]:
        """Recursively extract 'text' fields from JSON."""
        texts = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "text" and isinstance(v, str):
                    texts.append(v)
                else:
                    texts.extend(BaseAgent._extract_texts_recursive(v))
        elif isinstance(obj, list):
            for item in obj:
                texts.extend(BaseAgent._extract_texts_recursive(item))
        return texts
    
    def _orchestrate(self, user_text: str, tools: List[Dict]) -> Tuple[str, Dict, List[Dict]]:
        """Run A2A orchestration with intelligent question routing."""
        t0 = time.time()
        
        # 1. Fetch agent cards (parallel)
        candidates = self._fetch_all_agent_cards(self.a2a_agents)
        
        # 2. Group by batch
        batches: Dict[str, List] = {}
        for c in candidates:
            batch = c.get("batch") or self.a2a_default_batch
            batches.setdefault(batch, []).append(c)
        
        # 3. Select agents with specific questions per batch
        selected_all = []
        for batch_name, batch_candidates in batches.items():
            selected = self._select_agents_with_questions(batch_name, user_text, batch_candidates)
            for s in selected:
                s["batch"] = batch_name
            selected_all.extend(selected)
        
        # 4. Dedupe by name + limit
        seen = set()
        selected_final = []
        for s in selected_all:
            if s["name"] not in seen:
                seen.add(s["name"])
                selected_final.append(s)
        if self.a2a_max_selected > 0:
            selected_final = selected_final[:self.a2a_max_selected]
        
        routing_ms = int((time.time() - t0) * 1000)
        
        # 5. Call selected agents with their tailored questions
        subagent_results = []
        for s in selected_final:
            conf = next((c for c in candidates if c["name"] == s["name"]), None)
            if not conf:
                continue
            t1 = time.time()
            # Use the tailored question, not the original user text
            answer = self._call_a2a_agent(conf["url"], s["question"])
            elapsed_ms = int((time.time() - t1) * 1000)
            subagent_results.append({
                "name": conf["name"],
                "batch": s.get("batch"),
                "question": s["question"],  # The specific question we asked
                "reason": s.get("reason"),
                "answer": answer,           # The agent's response
                "elapsed_ms": elapsed_ms,
            })
        
        # 6. Build synthesis prompt with Q&A format
        qa_pairs = []
        for r in subagent_results:
            qa_pairs.append({
                "agent": r["name"],
                "question_asked": r["question"],
                "answer_received": r["answer"][:5000],
            })
        
        # Build synthesis prompt with fixed synthesis instruction + optional user instruction
        user_instruction_section = ""
        if self.instruction:
            user_instruction_section = f"""
ADDITIONAL AGENT INSTRUCTIONS (follow these alongside the synthesis rules):
{self.instruction}

---
"""
        
        synthesis_prompt = f"""{A2A_SYNTHESIS_INSTRUCTION}
{user_instruction_section}
USER'S ORIGINAL QUESTION:
{user_text}

INFORMATION GATHERED FROM SPECIALIZED AGENTS:
{json.dumps(qa_pairs, indent=2) if qa_pairs else "(No agents were consulted)"}

PROVIDE YOUR ANSWER:"""

        # 7. Final call with tools (instruction already included in synthesis_prompt)
        logger.info(f"Final synthesis with {len(tools)} tools, {len(qa_pairs)} agent Q&A pairs")
        response, answer = self._responses_create(synthesis_prompt, tools=tools, use_previous=True, include_instruction=False)
        self._last_response_id = response.id
        
        routing = {
            "routing_ms": routing_ms,
            "candidates": len(candidates),
            "batches": list(batches.keys()),
            "selected": [
                {"name": s["name"], "question": s["question"], "reason": s.get("reason")}
                for s in selected_final
            ],
        }
        
        return answer, routing, subagent_results
    
    # =========================================================================
    # Request Input Extraction
    # =========================================================================
    
    def _extract_input(self, request: ResponsesAgentRequest) -> str:
        """Extract input text from request."""
        parts = []
        for item in request.input:
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
        return "\n".join(parts)
    
    # =========================================================================
    # Main Entry Points
    # =========================================================================
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = []
        for event in self.predict_stream(request):
            if event.type == "response.output_item.done":
                outputs.append(event.item)
        return ResponsesAgentResponse(output=outputs, custom_outputs={})
    
    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        input_text = self._extract_input(request)
        tools = self._build_tools()
        
        self.last_routing = None
        self.last_subagent_results = None
        
        try:
            if self.a2a_agents:
                # Orchestration mode
                logger.info(f"Orchestration mode: {len(self.a2a_agents)} subagents configured")
                answer, routing, results = self._orchestrate(input_text, tools)
                self.last_routing = routing
                self.last_subagent_results = results
            else:
                # Simple mode (no subagents)
                logger.info(f"Simple mode: {len(tools)} tools")
                response, answer = self._responses_create(input_text, tools=tools, use_previous=True)
                self._last_response_id = response.id
            
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(
                    text=answer or "(No response)",
                    id=f"msg_{uuid4().hex[:8]}",
                ),
            )
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            if "500" in str(e):
                self._last_response_id = None
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(text=f"Error: {e}", id="msg_error"),
            )
    
    def reset_conversation(self):
        self._last_response_id = None


# =============================================================================
# Module Entry Point
# =============================================================================

def create_agent() -> BaseAgent:
    return BaseAgent()


agent = create_agent()
set_model(agent)
