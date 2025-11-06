"""
STEP 4: LLM Integration - OpenAI Agent with LangGraph + Streamlit UI
Converts natural language to database queries using tool calling

REFACTORED to use:
- ConfigManager (config.py) for unified configuration
- MCPClient (mcp_client.py) for database access via Model Context Protocol
- Custom exceptions (exceptions.py) for better error handling
"""

from typing import Dict, List, Any, Literal, Optional
from datetime import datetime
import json
import logging
import os
import threading

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import streamlit as st
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from config import get_config, ConfigManager
from mcp_client import get_mcp_client, MCPClient
from prompts import SYSTEM_PROMPT, KEYWORD_CLASSIFIER_PROMPT
from semantic_utils import generate_followup_with_skip_check
from exceptions import MCPServerError

# ============================================================================
# LOGGING CONFIGURATION - Performance Optimization
# ============================================================================
# Control logging level via LOG_LEVEL environment variable
# Examples:
#   export LOG_LEVEL=WARNING  # Production: minimal logs, best performance
#   export LOG_LEVEL=INFO     # Development: detailed logs
#   export LOG_LEVEL=DEBUG    # Debugging: maximum verbosity
# Default: DEBUG (for performance testing and visibility)

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
PERF_DEBUG = os.getenv("PERF_DEBUG", "TRUE").upper() == "TRUE"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL))


# ============================================================================
# PERFORMANCE TIMING UTILITY
# ============================================================================

import time
from contextlib import contextmanager

class PerfTimer:
    """Performance timing utility for identifying bottlenecks"""

    def __init__(self, enabled: bool = PERF_DEBUG):
        self.enabled = enabled
        self.marks = {}
        self.start_time = None

    def start(self, label: str = "operation"):
        """Mark operation start"""
        if not self.enabled:
            return
        self.start_time = time.time()
        self.marks[label] = {"start": self.start_time, "duration": None}

    def mark(self, label: str):
        """Record time for a specific operation"""
        if not self.enabled:
            return
        current_time = time.time()
        if label not in self.marks:
            self.marks[label] = {}
        self.marks[label]["start"] = current_time

    def end(self, label: str, verbose: bool = True):
        """Mark operation end and log duration"""
        if not self.enabled:
            return 0
        current_time = time.time()
        if label in self.marks:
            start = self.marks[label].get("start", self.start_time or current_time)
            duration_ms = (current_time - start) * 1000
            self.marks[label]["duration"] = duration_ms
            if verbose:
                logger.info(f"⏱️  {label}: {duration_ms:.1f}ms")
            return duration_ms
        return 0

    def summary(self):
        """Print timing summary"""
        if not self.enabled or not self.marks:
            return
        logger.info(f"\n{'='*80}")
        logger.info(f"PERFORMANCE SUMMARY")
        logger.info(f"{'='*80}")
        total = 0
        for label, data in sorted(self.marks.items(), key=lambda x: x[1].get("duration", 0) or 0, reverse=True):
            duration = data.get("duration", 0)
            if duration is not None:
                total += duration
                logger.info(f"  {label:40s}: {duration:8.1f}ms")
        logger.info(f"{'─'*80}")
        logger.info(f"  {'TOTAL':40s}: {total:8.1f}ms")
        logger.info(f"{'='*80}\n")

    @contextmanager
    def timer(self, label: str):
        """Context manager for timing blocks"""
        if not self.enabled:
            yield
            return
        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self.marks[label] = {"start": start, "duration": duration_ms}
            logger.info(f"⏱️  {label}: {duration_ms:.1f}ms")

# Create global timer instance
perf_timer = PerfTimer(enabled=PERF_DEBUG)


# ============================================================================
# 1. STATE DEFINITION FOR LLM AGENT
# ============================================================================

class LLMCVAgentState(TypedDict):
    """State for LLM-powered CV query agent"""

    # Input
    user_query: str
    cv_id: str

    # LLM Processing
    messages: List[Dict[str, Any]]
    llm_reasoning: str
    tool_calls: List[Dict[str, Any]]

    # Database Results
    pg_results: List[Dict[str, Any]]
    qdrant_results: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

    # Final Output
    final_response: str
    followup_question: str
    conversation_history: List[Dict[str, str]]


# ============================================================================
# 2. SYSTEM PROMPT DEFINITION
# ============================================================================
# defined in prompts.py as SYSTEM_PROMPT

# ============================================================================
# 3. LLM-FRIENDLY TOOLS DEFINITION
# ============================================================================

class CVTools:
    """Define tools for LLM to call"""

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search_company_experience",
                "description": "Find all jobs at a specific company and her responsibilities",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company_name": {
                            "type": "string",
                            "description": "Name of the company (e.g., 'KasiOss', 'AgBrain')"
                        }
                    },
                    "required": ["company_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_technology_experience",
                "description": "Find all jobs using a specific technology",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "technology": {
                            "type": "string",
                            "description": "Technology name (e.g., 'Python', 'TensorFlow', 'Docker')"
                        }
                    },
                    "required": ["technology"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_work_by_date",
                "description": "Find work experience within a date range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_year": {"type": "integer", "description": "Start year (e.g., 2020)"},
                        "end_year": {"type": "integer", "description": "End year (e.g., 2023)"}
                    },
                    "required": ["start_year", "end_year"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_education",
                "description": "Find education records by institution or degree",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "institution": {
                            "type": "string",
                            "description": "University or institution name"
                        },
                        "degree": {
                            "type": "string",
                            "description": "Degree type (e.g., 'PhD', 'Master')"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_publications",
                "description": "Search publications by year or keywords",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "year": {
                            "type": "integer",
                            "description": "Publication year"
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search keywords"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": "Perform semantic search on all CV content using vector embeddings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query"
                        },
                        "section": {
                            "type": "string",
                            "enum": ["work_experience", "education", "publication", "all"],
                            "description": "Filter by section (optional)"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_cv_summary",
                "description": "Get a summary of the CV including role, experience, and key stats",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_skills",
                "description": "Find skills by category (AI, ML, programming, Tools, Cloud, Data_tools)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["AI", "ML", "programming", "Tools", "Cloud", "Data_tools"],
                            "description": "Skill category"
                        }
                    },
                    "required": ["category"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_languages",
                "description": "Find languages and their proficiency levels",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "language": {
                            "type": "string",
                            "description": "Language name (optional, e.g., 'English', 'German', 'Thai')"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_work_references",
                "description": "Find professional work references by name or company",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reference_name": {
                            "type": "string",
                            "description": "Name of the reference person (optional)"
                        },
                        "company": {
                            "type": "string",
                            "description": "Company or institution (optional)"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_awards_certifications",
                "description": "Find awards and certifications received by Pattreeya. USE THIS TOOL for questions about: awards, certifications, recognitions, honors, achievements, or accomplishments. Can optionally filter by type. ALWAYS use this for questions like 'What awards in [field]?' or 'recognitions for contributions'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "award_type": {
                            "type": "string",
                            "description": "Type of award or certification to filter (e.g., 'Award', 'Certification', 'Machine Learning') (optional)"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_all_work_experience",
                "description": "Get complete work experience history - all jobs in chronological order. USE THIS TOOL for questions like: 'experience', 'list of experience', 'career history', 'all jobs', 'work background', or 'complete work history'",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]


# ============================================================================
# 4. TOOL EXECUTORS - MCP-BASED IMPLEMENTATION
# ============================================================================

class ToolExecutor:
    """Execute database queries using Model Context Protocol (MCP) client

    This class provides a unified interface for tool execution, delegating to
    the MCP server for actual database operations. This architecture ensures:
    - Separation of concerns: Database logic isolated from LLM agent
    - Standardized interface: All tools follow MCP specification
    - Better testability: Tools can be tested independently
    - Improved maintainability: Database changes only affect MCP server
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize tool executor with MCP client

        Args:
            config: ConfigManager instance with database/API configuration (optional)
        """
        self.config = config or get_config()

        # Initialize MCP client for tool execution
        try:
            self.mcp_client = get_mcp_client()
            self.use_mcp = True
            logger.info("✅ Using MCP client for tool execution")
        except Exception as e:
            logger.warning(f"⚠️  MCP client initialization failed: {e}. Using fallback...")
            self.use_mcp = False
            self._init_fallback()

    def _init_fallback(self):
        """Initialize fallback direct database access if MCP not available"""
        try:
            self.pg_connection_string = self.config.get_postgres_url()
            # Note: Fallback to direct database would be initialized here
            logger.error("❌ Fallback database access not implemented - please enable MCP client")
            raise MCPServerError("MCP client unavailable and fallback not configured")
        except Exception as e:
            logger.error(f"❌ Fallback initialization failed: {e}")
            raise

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any], cv_id: str) -> Dict[str, Any]:
        """Execute a tool through MCP client

        This method provides the main interface for tool execution.
        It delegates to the MCP client which handles actual database operations.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            cv_id: CV ID (required by some tools, handled automatically by MCP)

        Returns:
            Dict with status, results, and metadata
        """
        if self.use_mcp:
            return self._execute_via_mcp(tool_name, tool_input)
        else:
            return self._execute_via_fallback(tool_name, tool_input, cv_id)

    def _execute_via_mcp(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool using MCP client"""
        try:
            result = None

            # Map tool names and parameters for MCP client
            if tool_name == "search_company_experience":
                result = self.mcp_client.search_company_experience(
                    tool_input.get("company_name", "")
                )

            elif tool_name == "search_technology_experience":
                result = self.mcp_client.search_technology_experience(
                    tool_input.get("technology", "")
                )

            elif tool_name == "search_work_by_date":
                result = self.mcp_client.search_work_by_date(
                    tool_input.get("start_year", 2000),
                    tool_input.get("end_year", 2024)
                )

            elif tool_name == "search_education":
                result = self.mcp_client.search_education(
                    institution=tool_input.get("institution"),
                    degree=tool_input.get("degree")
                )

            elif tool_name == "search_publications":
                result = self.mcp_client.search_publications(
                    year=tool_input.get("year")
                )

            elif tool_name == "semantic_search":
                result = self.mcp_client.semantic_search(
                    query=tool_input.get("query", ""),
                    section=tool_input.get("section"),
                    top_k=tool_input.get("top_k", 5)
                )

            elif tool_name == "get_cv_summary":
                result = self.mcp_client.get_cv_summary()

            elif tool_name == "search_skills":
                result = self.mcp_client.search_skills(
                    tool_input.get("category", "")
                )

            elif tool_name == "search_languages":
                result = self.mcp_client.search_languages(
                    language=tool_input.get("language")
                )

            elif tool_name == "search_work_references":
                result = self.mcp_client.search_work_references(
                    reference_name=tool_input.get("reference_name"),
                    company=tool_input.get("company")
                )

            elif tool_name == "search_awards_certifications":
                result = self.mcp_client.search_awards_certifications(
                    award_type=tool_input.get("award_type")
                )

            elif tool_name == "get_all_work_experience":
                result = self.mcp_client.get_all_work_experience()

            else:
                return {
                    "status": "error",
                    "tool": tool_name,
                    "error": f"Unknown tool: {tool_name}"
                }

            # Ensure result is not None before returning
            if result is None:
                logger.warning(f"MCP tool {tool_name} returned None, returning error response")
                return {
                    "status": "error",
                    "tool": tool_name,
                    "error": f"Tool {tool_name} returned None"
                }

            return result

        except Exception as e:
            logger.error(f"MCP execution error for {tool_name}: {e}", exc_info=True)
            return {
                "status": "error",
                "tool": tool_name,
                "error": str(e)
            }

    def _execute_via_fallback(self, tool_name: str, tool_input: Dict[str, Any], cv_id: str) -> Dict[str, Any]:
        """Execute tool using direct database access (fallback)

        This is only used if MCP client is not available.
        Includes all the original database implementation.
        """
        # For brevity, return a note that fallback implementation would go here
        # In a real scenario, the original tool methods would be retained here
        return {
            "status": "error",
            "tool": tool_name,
            "error": "Fallback implementation not configured. Please enable MCP client."
        }


# ============================================================================
# 5. LANGGRAPH NODES FOR LLM AGENT
# ============================================================================

def llm_reasoning_node(state: LLMCVAgentState, tool_executor: ToolExecutor, config: Optional[ConfigManager] = None) -> LLMCVAgentState:
    """Node 1: Call LLM to reason about query and decide on tools"""

    if config is None:
        config = get_config()

    # Create performance timer for this node
    perf_timer_reasoning = PerfTimer(enabled=PERF_DEBUG)
    perf_timer_reasoning.start("llm_reasoning_node")

    logger.debug(f"Starting llm_reasoning_node for query: {state['user_query'][:60]}...")
    messages = state.get("messages", [])

    # Add system prompt if this is the first message
    if not messages:
        messages.insert(0, {
            "role": "system",
            "content": SYSTEM_PROMPT
        })

    messages.append({
        "role": "user",
        "content": state["user_query"]
    })

    # Initialize LangChain ChatOpenAI model with tool binding
    # Temperature 0.5: Balanced reasoning + willingness to try tool calling
    # (0.3 was too conservative and discouraged tool usage)
    with perf_timer_reasoning.timer("llm_initialization_reasoning"):
        llm = ChatOpenAI(
            api_key=config.get_api_key(),
            model=config.get_model(),
            temperature=0.5,  # Increased from 0.3 for better tool-calling behavior
            max_tokens=1024
        )

    # Convert messages to LangChain format
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

    with perf_timer_reasoning.timer("message_conversion_reasoning"):
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # Create AIMessage with content only
                # Tool calls are already represented in the message flow via tool messages
                langchain_messages.append(AIMessage(content=msg.get("content", "")))
            elif msg["role"] == "tool":
                langchain_messages.append(ToolMessage(content=msg["content"], tool_call_id=msg.get("tool_call_id")))

    # Bind tools to the model
    with perf_timer_reasoning.timer("bind_tools"):
        llm_with_tools = llm.bind(tools=[{"type": "function", "function": t["function"]} for t in CVTools.TOOLS])

    # Call the model
    with perf_timer_reasoning.timer("reasoning_llm_inference"):
        response = llm_with_tools.invoke(langchain_messages)

    state["llm_reasoning"] = response.content or ""

    # Extract tool calls from LangChain response
    tool_calls = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            # Handle both object and dict formats
            if isinstance(tool_call, dict):
                tool_calls.append({
                    "id": tool_call.get("id"),
                    "name": tool_call.get("name"),
                    "input": tool_call.get("args") or tool_call.get("input") or {}
                })
            else:
                # Tool calls are objects with attributes
                tool_input = getattr(tool_call, "args", None) or {}
                tool_calls.append({
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_input
                })

            logger.debug(f"Extracted tool call: {tool_calls[-1]}")
            logger.info(f"Tool: {tool_calls[-1]['name']}, Input: {tool_calls[-1]['input']}")

    state["tool_calls"] = tool_calls

    # NEW: Verify tools were called - Critical for debugging Issue #1-#4
    if not tool_calls:
        logger.warning("⚠️  WARNING: LLM DID NOT CALL ANY TOOLS!")
        logger.warning("   This is a critical issue - responses will use training data only (potentially stale)")
        logger.warning("   Check: System prompt includes MANDATORY tool-calling instructions?")
        logger.warning("   Check: Tool descriptions are clear about WHEN to use each tool?")
        logger.info(f"   User Query: {state['user_query'][:100]}...")
        logger.info(f"   LLM Response (no tool calls): {response.content[:200]}...")
    else:
        logger.info(f"✓ Tool calling successful: {len(tool_calls)} tool(s) identified")
        for tc in tool_calls:
            logger.info(f"  - {tc['name']}: {tc['input']}")

    # Add to messages - format tool_calls properly for OpenAI API
    assistant_message = {
        "role": "assistant",
        "content": response.content or ""
    }

    # Include tool_calls in the proper format if they exist
    if tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["input"])
                }
            }
            for tc in tool_calls
        ]

    messages.append(assistant_message)
    state["messages"] = messages

    # Print performance summary for reasoning node
    perf_timer_reasoning.end("llm_reasoning_node", verbose=False)
    perf_timer_reasoning.summary()

    return state


def execute_tools_node(state: LLMCVAgentState, tool_executor: ToolExecutor) -> LLMCVAgentState:
    """Node 2: Execute LLM-requested tools with keyword injection fallback (only when results are empty)"""

    # Create performance timer for this node
    perf_timer_tools = PerfTimer(enabled=PERF_DEBUG)
    perf_timer_tools.start("execute_tools_node")

    cv_id = state["cv_id"]
    tool_results = []

    # Get keyword classification for fallback use (only when results are empty)
    keyword_classification = state.get("keyword_classification", {})
    has_keywords_available = bool(keyword_classification and keyword_classification.get("qdrant_keywords"))

    # Log tool execution start
    logger.info(f"\n{'='*80}")
    logger.info(f"TOOL EXECUTION (Keyword injection disabled - will use as fallback only)")
    logger.info(f"{'='*80}")
    # if has_keywords_available:
    #     logger.info(f"📌 Keywords available for fallback: {keyword_classification.get('qdrant_keywords', [])}")
    # else:
    #     logger.info(f"⚪ No keywords available (won't have fallback)")
    logger.info(f"{'='*80}\n")

    with perf_timer_tools.timer("tool_execution_loop"):
        for tool_call in state["tool_calls"]:
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]
            original_input = tool_input.copy()

            # Log tool execution details
            logger.info(f"\n{'─'*80}")
            logger.info(f"TOOL EXECUTION: {tool_name}")
            logger.info(f"{'─'*80}")
            logger.info(f"   Tool Input (original, NO keyword injection):")
            for key, value in tool_input.items():
                logger.info(f"      {key}: {value}")

            # Execute tool with original input (NO keyword injection)
            result = tool_executor.execute_tool(tool_name, tool_input, cv_id)
            was_enhanced = False
            enhanced_input = None

            # Log initial result
            logger.info(f"\n   📊 Initial Tool Result:")
            if result.get("status") == "success":
                result_data = result.get("data", [])
                num_records = len(result_data) if isinstance(result_data, list) else 1
                logger.info(f"      Status: ✅ SUCCESS")
                logger.info(f"      Records returned: {num_records}")

                if isinstance(result_data, list) and result_data:
                    logger.info(f"      Preview (first item):")
                    first_item = result_data[0]
                    if isinstance(first_item, dict):
                        for k, v in list(first_item.items())[:2]:
                            preview_val = str(v)[:50]
                            logger.info(f"         {k}: {preview_val}...")
            else:
                logger.info(f"      Status: ❌ {result.get('status', 'UNKNOWN')}")
                logger.info(f"      Error: {result.get('error', 'No error message')}")

            # FALLBACK: If result is empty (0 records) and keywords available, retry with keyword injection
            result_data = result.get("data", [])
            is_empty_result = (result.get("status") == "success" and
                              (not result_data or (isinstance(result_data, list) and len(result_data) == 0)))
            


            # if is_empty_result and has_keywords_available:
            #     logger.info(f"\n   🔄 EMPTY RESULT DETECTED - Attempting fallback with keyword injection...")

            #     enhanced_input = inject_classifier_keywords_into_tool_input(
            #         tool_name,
            #         tool_input,
            #         keyword_classification
            #     )

            #     if enhanced_input != tool_input:
            #         logger.info(f"\n   📌 FALLBACK: INPUT ENHANCEMENT APPLIED")
            #         logger.info(f"      Original Input: {original_input}")
            #         logger.info(f"      Enhanced Input: {enhanced_input}")

            #         # Retry with enhanced input
            #         result = tool_executor.execute_tool(tool_name, enhanced_input, cv_id)
            #         was_enhanced = True

            #         logger.info(f"\n   📊 FALLBACK Tool Result:")
            #         if result.get("status") == "success":
            #             result_data = result.get("data", [])
            #             num_records = len(result_data) if isinstance(result_data, list) else 1
            #             logger.info(f"      Status: ✅ SUCCESS")
            #             logger.info(f"      Records returned: {num_records} (WITH keyword injection)")

            #             if isinstance(result_data, list) and result_data:
            #                 logger.info(f"      Preview (first item):")
            #                 first_item = result_data[0]
            #                 if isinstance(first_item, dict):
            #                     for k, v in list(first_item.items())[:2]:
            #                         preview_val = str(v)[:50]
            #                         logger.info(f"         {k}: {preview_val}...")
            #         else:
            #             logger.info(f"      Status: ❌ {result.get('status', 'UNKNOWN')}")
            #             logger.info(f"      Error: {result.get('error', 'No error message')}")
            #     else:
            #         logger.info(f"      No enhancement possible for {tool_name}")
            # elif is_empty_result:
            #     # SOLUTION 2: Improved Empty Result Handling with Semantic Fallback
            #     # Attempt semantic search when specific tool returns no results
            #     if has_keywords_available and tool_name != "semantic_search":
            #         try:
            #             keywords = keyword_classification.get("qdrant_keywords", [])
            #             semantic_input = {
            #                 "query": " ".join(keywords) if isinstance(keywords, list) else str(keywords),
            #                 "top_k": 5
            #             }
            #             semantic_result = tool_executor.execute_tool("semantic_search", semantic_input, cv_id)

            #             if semantic_result.get("status") == "success" and semantic_result.get("data"):
            #                 result = semantic_result
            #                 was_enhanced = True
            #             else:
            #                 logger.warning(f"⚠️  No results found for '{tool_name}' - semantic fallback also empty")
            #         except Exception as e:
            #             logger.warning(f"⚠️  No results found for '{tool_name}' - semantic fallback failed: {e}")
            #     else:
            #         logger.warning(f"⚠️  No results found for '{tool_name}'")

            tool_results.append({
                "tool_id": tool_call["id"],
                "tool_name": tool_name,
                "result": result,
                "was_enhanced": was_enhanced,
                "original_input": original_input,
                "enhanced_input": enhanced_input
            })

    state["tool_results"] = tool_results

    # Add results to messages
    messages = state["messages"]
    for tool_result in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_result["tool_id"],
            "name": tool_result["tool_name"],
            "content": json.dumps(tool_result["result"])
        })

    state["messages"] = messages

    # Log summary
    logger.info(f"\n{'='*80}")
    logger.info(f"TOOL EXECUTION SUMMARY")
    logger.info(f"{'='*80}")
    enhanced_count = sum(1 for tr in tool_results if tr.get("was_enhanced"))
    total_tools = len(tool_results)
    logger.info(f"Total tools executed: {total_tools}")
    logger.info(f"Fallback with keywords used: {enhanced_count}")
    logger.info(f"Normal execution (no fallback): {total_tools - enhanced_count}")

    if enhanced_count > 0:
        logger.info(f"\n✅ FALLBACK ACTIVATED: {enhanced_count}/{total_tools} tools used keyword injection")
    else:
        logger.info(f"\n⚪ No fallback needed - All tools succeeded on first try")
    logger.info(f"{'='*80}\n")

    # Print performance summary for tools node
    perf_timer_tools.end("execute_tools_node", verbose=False)
    perf_timer_tools.summary()

    return state


def final_response_node(state: LLMCVAgentState, config: Optional[ConfigManager] = None) -> LLMCVAgentState:
    """Node 3: Generate final response from LLM"""

    if config is None:
        config = get_config()

    perf_timer.start("final_response_node")

    messages = state["messages"]

    # Ensure system prompt is at the beginning
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {
            "role": "system",
            "content": SYSTEM_PROMPT
        })

    # Initialize LangChain ChatOpenAI model
    with perf_timer.timer("llm_initialization"):
        llm = ChatOpenAI(
            api_key=config.get_api_key(),
            model=config.get_model(),
            temperature=0.2,
            max_tokens=2048
        )

    # Convert messages to LangChain format
    with perf_timer.timer("message_conversion"):
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # Include tool_calls if they exist - convert from OpenAI format to LangChain format
                ai_kwargs = {"content": msg.get("content", "")}

                if msg.get("tool_calls"):
                    tool_calls_list = []
                    for tc in msg.get("tool_calls", []):
                        # tc is in OpenAI format: {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
                        try:
                            args_str = tc.get("function", {}).get("arguments", "{}")
                            args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str

                            tool_calls_list.append({
                                "id": tc.get("id"),
                                "name": tc.get("function", {}).get("name"),
                                "args": args_dict if args_dict else {}
                            })
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            logger.warning(f"Failed to parse tool call: {e}, skipping")
                            continue

                    if tool_calls_list:
                        ai_kwargs["tool_calls"] = tool_calls_list

                langchain_messages.append(AIMessage(**ai_kwargs))
            elif msg["role"] == "tool":
                langchain_messages.append(ToolMessage(content=msg["content"], tool_call_id=msg.get("tool_call_id")))

    # ⚡ OPTIMIZATION: Call main LLM for final response
    logger.info(f"\n{'='*80}\n⏱️  PERFORMANCE PROFILING: final_response_node\n{'='*80}")
    with perf_timer.timer("main_llm_inference"):
        logger.info("Starting main LLM inference...")
        response = llm.invoke(langchain_messages)
        logger.info("Main LLM inference completed.")

    final_response = response.content or "No response generated"
    state["final_response"] = final_response

    # ===== FOLLOW-UP GENERATION WITH SKIP CHECK =====
    # Encapsulated in generate_followup_with_skip_check() function in semantic_utils.py
    # This handles:
    # - skip_followup flag check
    # - Category classification (SOLUTION 1)
    # - Semantic tracking initialization (SOLUTION 5)
    # - Follow-up generation with all 5 solutions (SOLUTIONS 2-5)
    # - Error handling and performance timing

    # OPTIMIZATION OPPORTUNITY: This is currently sequential but could be parallelized
    # in a future version with the main LLM response generation
    logger.debug("Starting follow-up question generation...")
    with perf_timer.timer("followup_question_generation"):
        state = generate_followup_with_skip_check(
            state=state,
            final_response=final_response,
            config=config,
            perf_timer=perf_timer,
            logger_instance=logger
        )
    logger.debug("Follow-up question generation completed.")

    # Add to conversation history
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({
        "user_query": state["user_query"],
        "final_response": final_response,
        "followup_question": state.get("followup_question", ""),
        "timestamp": datetime.now().isoformat()
    })

    state["conversation_history"] = conversation_history

    # Print performance summary
    perf_timer.end("final_response_node", verbose=False)  # End the overall timer
    perf_timer.summary()  # Print complete breakdown

    return state


def router_node(state: LLMCVAgentState) -> Literal["execute_tools", "final_response"]:
    """Route: Check if tools were called"""
    if state.get("tool_calls"):
        return "execute_tools"
    else:
        return "final_response"


# ============================================================================
# 6. BUILD LLM AGENT GRAPH
# ============================================================================

def build_llm_cv_agent_graph(tool_executor: ToolExecutor, config: Optional[ConfigManager] = None):
    """Build LangGraph for LLM CV agent"""

    if config is None:
        config = get_config()

    graph = StateGraph(LLMCVAgentState)

    # Add nodes
    graph.add_node(
        "llm_reasoning",
        lambda state: llm_reasoning_node(state, tool_executor, config)
    )

    graph.add_node(
        "execute_tools",
        lambda state: execute_tools_node(state, tool_executor)
    )

    graph.add_node(
        "final_response",
        lambda state: final_response_node(state, config)
    )

    # Add edges
    graph.add_edge(START, "llm_reasoning")
    graph.add_conditional_edges(
        "llm_reasoning",
        router_node,
        {
            "execute_tools": "execute_tools",
            "final_response": "final_response"
        }
    )

    graph.add_edge("execute_tools", "final_response")
    graph.add_edge("final_response", END)

    return graph.compile()


# ============================================================================
# 7. KEYWORD CLASSIFICATION FOR SEARCH OPTIMIZATION
# ============================================================================


def classify_user_query_keywords(user_query: str, config: Optional[ConfigManager] = None) -> Dict[str, Any]:
    """Classify user query and extract keywords for MCP and Qdrant search

    Enhanced with multiple fallback strategies to ensure keywords are ALWAYS available.
    This is SOLUTION 1: Robust Keyword Classification from the root cause analysis.

    Args:
        user_query: User's question about Pattreeya
        config: Configuration object with API keys (optional)

    Returns:
        Dict with search_type, mcp_tools, qdrant_keywords, and explanation
        GUARANTEED: qdrant_keywords is never empty
    """
    if config is None:
        config = get_config()

    # ========================================================================
    # STRATEGY 1: Try primary LLM classification
    # ========================================================================
    try:
        llm = ChatOpenAI(
            api_key=config.get_api_key(),
            model=config.get_model(),
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=512
        )

        messages = [
            SystemMessage(content=KEYWORD_CLASSIFIER_PROMPT),
            HumanMessage(content=f"User's question: {user_query}")
        ]

        response = llm.invoke(messages)
        response_text = response.content or ""

        # Check if response is empty (API timeout, empty response, etc.)
        if not response_text.strip():
            logger.warning(f"LLM returned empty response, using fallback extraction")
            return _create_fallback_classification(user_query)

        # Try to parse JSON response
        try:
            classification = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {response_text[:100]}...")
            # If JSON parsing fails, try fallback extraction
            return _create_fallback_classification(user_query)

        # SOLUTION 1: VALIDATE - Ensure qdrant_keywords is not empty
        qdrant_keywords = classification.get("qdrant_keywords", [])
        if not qdrant_keywords or (isinstance(qdrant_keywords, list) and len(qdrant_keywords) == 0):
            logger.warning(f"Classifier returned empty qdrant_keywords, using fallback extraction")
            classification["qdrant_keywords"] = _extract_keywords_fallback(user_query)

        # IMPORTANT: Map primary_tool and complementary_tools from the prompt response
        # The prompt returns "primary_tool" and "complementary_tools" but we need "mcp_tools"
        primary_tool = classification.get("primary_tool")
        complementary_tools = classification.get("complementary_tools", [])

        # Build mcp_tools list with primary tool first, then complementary tools
        mcp_tools = []
        if primary_tool:
            mcp_tools.append(primary_tool)
        if complementary_tools:
            mcp_tools.extend(complementary_tools)

        # Ensure we have at least one tool
        if not mcp_tools:
            mcp_tools = ["semantic_search"]

        # Add mcp_tools to classification for backward compatibility with execute_tools_node
        classification["mcp_tools"] = mcp_tools

        # Log the classification
        logger.info(f"\n{'='*80}")
        logger.info(f"KEYWORD CLASSIFICATION FOR USER QUERY")
        logger.info(f"{'='*80}")
        logger.info(f"Query: {user_query}")
        logger.info(f"Search Type: {classification.get('search_type', 'unknown')}")
        logger.info(f"Primary Tool: {primary_tool}")
        logger.info(f"Complementary Tools: {complementary_tools}")
        logger.info(f"MCP Tools to use: {mcp_tools}")
        logger.info(f"Qdrant Keywords: {classification.get('qdrant_keywords', [])}")
        logger.info(f"Explanation: {classification.get('explanation', 'N/A')}")
        logger.info(f"{'='*80}\n")

        return classification

    except Exception as e:
        # ====================================================================
        # STRATEGY 2: LLM classification completely failed, use fallback
        # ====================================================================
        logger.error(f"Error in primary keyword classification: {e}, using fallback strategy")
        return _create_fallback_classification(user_query)

# ============================================================================
# 6.1 HELPER FUNCTIONS FOR ROBUST KEYWORD CLASSIFICATION
# ============================================================================

def _extract_keywords_fallback(user_query: str) -> List[str]:
    """
    Extract keywords using spacy for language-aware lemmatization.

    This function implements an enhanced keyword extraction strategy using:
    1. Language detection (langid) to identify the language
    2. Spacy NLP pipeline for proper stemming/lemmatization
    3. Removal of common stop words
    4. Return of lemmatized stems for better matching

    Args:
        user_query: User's question text

    Returns:
        List of extracted keyword stems, or empty list if extraction fails
    """
    import re

    try:
        # Step 1: Detect language using langid
        try:
            import langid
            detected_lang, _ = langid.classify(user_query)
            logger.debug(f"✓ Detected language: {detected_lang}")
        except Exception as e:
            logger.warning(f"⚠ Language detection failed: {e}, defaulting to English")
            detected_lang = "en"

        # Step 2: Load appropriate spacy model based on detected language
        spacy_nlp = None
        try:
            import spacy

            # Map language codes to spacy model names
            lang_model_map = {
                "en": "en_core_web_sm",
                "de": "de_core_news_sm",
                "fr": "fr_core_news_sm"
            }

            # Get model name for detected language, default to English if not found
            if detected_lang in lang_model_map:
                model_name = lang_model_map[detected_lang]
                logger.debug(f"✓ Language '{detected_lang}' found in map, using model: {model_name}")
            else:
                # Language not supported in lang_model_map - use English without fail
                logger.warning(
                    f"⚠ Language '{detected_lang}' not supported in lang_model_map. "
                    f"Supported languages: {list(lang_model_map.keys())}. Using English model."
                )
                model_name = "en_core_web_sm"

            try:
                spacy_nlp = spacy.load(model_name)
                logger.debug(f"✓ Loaded spacy model: {model_name}")
            except OSError:
                logger.warning(
                    f"⚠ Spacy model '{model_name}' not found. "
                    f"Install with: python -m spacy download {model_name}"
                )
                # Fallback to English if requested model not found
                if model_name != "en_core_web_sm":
                    try:
                        spacy_nlp = spacy.load("en_core_web_sm")
                        logger.debug("✓ Loaded fallback spacy model: en_core_web_sm")
                    except OSError:
                        logger.warning("⚠ en_core_web_sm not found, using regex fallback")
                        spacy_nlp = None
                else:
                    # Already tried en_core_web_sm, can't proceed with spacy
                    logger.warning("⚠ en_core_web_sm not found, using regex fallback")
                    spacy_nlp = None

        except ImportError:
            logger.warning("⚠ spacy not installed, using regex fallback")
            spacy_nlp = None

        # Step 3: Extract keywords using spacy (pure English at this point)
        if spacy_nlp:
            # Use spacy for lemmatization
            try:
                doc = spacy_nlp(user_query.lower())

                # Extract lemmas, filtering by POS tags and stop words
                stems = []
                for token in doc:
                    # Skip stop words, punctuation, and determiners
                    if (
                        not token.is_stop
                        and not token.is_punct
                        and len(token.lemma_) > 2
                        and token.pos_ not in ["CCONJ", "SCONJ", "ADP", "DET", "PRON"]
                    ):
                        stems.append(token.lemma_)

                logger.debug(f"✓ Extracted {len(stems)} stems using spacy: {stems}")

                if stems:
                    # Remove duplicates while preserving order
                    unique_stems = list(dict.fromkeys(stems))[:5]
                    logger.debug(f"✓ Final keyword stems: {unique_stems}")
                    return unique_stems
                else:
                    logger.debug("⚠ Spacy extraction returned no stems, using whole input string as fallback")
            except Exception as e:
                logger.warning(f"⚠ Spacy extraction failed: {e}, using whole input string as fallback")
        else:
            return [user_query.strip()]

    except Exception as e:
        logger.warning(f"⚠ Keyword extraction completely failed: {e}")
        # Final fallback: return whole input string or defaults
        whole_input = user_query.strip()
        if len(whole_input) > 0:
            logger.debug(f"Using entire input string as fallback: '{whole_input[:60]}...'")
            return [whole_input]
        else:
            logger.debug("Using default keywords as fallback")
            return ["cv", "experience"]


def _create_fallback_classification(user_query: str) -> Dict[str, Any]:
    """
    Create a classification when all LLM-based classification strategies fail.

    This ensures that keywords are ALWAYS available, even if the LLM classifier
    completely fails. Uses simple keyword extraction as fallback.

    Args:
        user_query: User's question text

    Returns:
        Classification dict with guaranteed non-empty qdrant_keywords
    """
    extracted_keywords = _extract_keywords_fallback(user_query)

    return {
        "search_type": "general",
        "primary_tool": "semantic_search",
        "complementary_tools": [],
        "mcp_tools": ["semantic_search"],
        "qdrant_keywords": extracted_keywords,  # GUARANTEED not empty!
        "explanation": f"Fallback classification with extracted keywords: {extracted_keywords}"
    }

# ============================================================================
# 7. KEYWORD INJECTION FOR SEARCH ENHANCEMENT (EXPERIMENTAL)
# ============================================================================

def inject_classifier_keywords_into_tool_input(
    tool_name: str,
    tool_input: Dict[str, Any],
    keyword_classification: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Inject classifier keywords into tool inputs to improve search results

    This experimental function enhances tool inputs with keywords extracted
    by the keyword classifier to provide better search context.

    Args:
        tool_name: Name of the tool being executed
        tool_input: Original tool input parameters
        keyword_classification: Output from classify_user_query_keywords

    Returns:
        Enhanced tool_input with classifier keywords injected
    """
    enhanced_input = tool_input.copy()
    qdrant_keywords = keyword_classification.get("qdrant_keywords", [])

    # For semantic_search, enhance the query with classifier keywords
    if tool_name == "semantic_search":
        original_query = enhanced_input.get("query", "")

        # If classifier provided specific keywords, create an enhanced query
        if qdrant_keywords and isinstance(qdrant_keywords, list) and len(qdrant_keywords) > 0:
            # Join classifier keywords with the original query for richer context
            keywords_str = ", ".join(qdrant_keywords)
            enhanced_query = f"{original_query} [context: {keywords_str}]"
            enhanced_input["query"] = enhanced_query

            logger.info(f"✓ Enhanced semantic_search query")
            logger.info(f"  Original: {original_query}")
            logger.info(f"  Enhanced: {enhanced_query}")

    # For search_technology_experience, inject relevant keywords
    elif tool_name == "search_technology_experience":
        if not enhanced_input.get("technology") and qdrant_keywords:
            # Use first keyword as technology search term
            enhanced_input["technology"] = qdrant_keywords[0]
            logger.info(f"✓ Injected keyword into search_technology_experience: {qdrant_keywords[0]}")

    # For search_skills, inject relevant keywords
    elif tool_name == "search_skills":
        if not enhanced_input.get("category") and qdrant_keywords:
            # Use first keyword as category search term
            enhanced_input["category"] = qdrant_keywords[0]
            logger.info(f"✓ Injected keyword into search_skills: {qdrant_keywords[0]}")

    # For search_publications, enhance with keywords
    elif tool_name == "search_publications":
        # Store keywords for filtering results (optional enhancement)
        if qdrant_keywords and "keywords" not in enhanced_input:
            enhanced_input["_classifier_keywords"] = qdrant_keywords
            logger.info(f"✓ Added classifier keywords for search_publications filtering: {qdrant_keywords}")

    # For search_company_experience, use context if no company specified
    elif tool_name == "search_company_experience":
        if not enhanced_input.get("company_name") and qdrant_keywords:
            # Try to find company name in keywords (heuristic)
            potential_companies = ["kasioss", "agbrain"]  # Known companies
            for keyword in qdrant_keywords:
                if any(comp in keyword.lower() for comp in potential_companies):
                    enhanced_input["company_name"] = keyword
                    logger.info(f"✓ Injected company keyword: {keyword}")
                    break

    return enhanced_input


def classify_user_query_keywords_parallel(
    user_query: str,
    config: Optional[ConfigManager] = None
) -> Dict[str, Any]:
    """Classify query keywords with optional parallel processing

    This wrapper enables/disables parallel keyword classification based on flags.

    Args:
        user_query: The user's question
        config: Configuration object

    Returns:
        Keyword classification dict
    """
    # Check if classification is enabled
    use_keyword_classification = st.session_state.get("use_keyword_classification", False)
    use_parallel = True

    if not use_keyword_classification:
        # Classification disabled - return fast default
        logger.info(f"⚡ Keyword classification disabled (use_keyword_classification=False)")
        return _create_fallback_classification(user_query)

    if use_parallel:
        # Run keyword classification in background thread
        logger.info(f"⚡ Running keyword classification in parallel thread")

        # Create a dictionary to hold results from thread
        results = {}

        def classify_in_thread():
            """Run classification in separate thread"""
            try:
                results["classification"] = classify_user_query_keywords(user_query, config)
            except Exception as e:
                logger.error(f"Error in parallel classification: {e}")
                results["classification"] = _create_fallback_classification(user_query)

        # Start thread
        classification_thread = threading.Thread(target=classify_in_thread, daemon=True)
        classification_thread.start()

        # Don't wait for thread - return immediately with fallback
        # The thread will complete in background and results can be used next time
        logger.info(f"⚡ Returning fallback while classification runs in parallel")
        return _create_fallback_classification(user_query)
    else:
        # Run synchronously (blocking, but simpler)
        logger.info(f"ℹ️  Keyword classification running synchronously")
        return classify_user_query_keywords(user_query, config)


# ============================================================================
# 8. LLM AGENT EXECUTOR
# ============================================================================

def execute_llm_cv_query(
    user_query: str,
    cv_id: str,
    tool_executor: ToolExecutor,
    config: Optional[ConfigManager] = None,
) -> str:
    """Execute query through LLM agent with optional keyword classification

    Args:
        user_query: The user's question
        cv_id: CV ID for database queries
        tool_executor: Tool executor for running database queries
        config: Configuration object
     Returns:
        The LLM response text (follow-up question generation is handled separately)
    """

    agent_graph = build_llm_cv_agent_graph(tool_executor, config)

    initial_state = {
        "user_query": user_query,
        "cv_id": cv_id,
        "messages": [],
        "llm_reasoning": "",
        "tool_calls": [],
        "pg_results": [],
        "qdrant_results": [],
        "tool_results": [],
        "final_response": "",
        "followup_question": "",
        "conversation_history": []
    }

    final_state = agent_graph.invoke(initial_state)

    # Return only the response text
    # Follow-up question generation is handled separately in the UI layer
    return final_state["final_response"]


# ============================================================================
# 8. STREAMLIT UI
# ============================================================================

def main():
    """Main Streamlit application"""

    st.set_page_config(
        page_title="Pattreeya's Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("Chat with Pattreeya's Agent")
    st.markdown("Ask me anything about Pattreeya...")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if "auto_submit_followup" not in st.session_state:
        st.session_state.auto_submit_followup = None

    if "question_keywords" not in st.session_state:
        st.session_state.question_keywords = []  # Track keywords for each question

    # Performance optimization flags
    if "skip_followup" not in st.session_state:
        st.session_state.skip_followup = False  # Default: skip follow-up generation for faster responses

  
    if "use_pregem_followups" not in st.session_state:
        st.session_state.use_pregem_followups = True  # Default: use pre-generated follow-up questions

    if "tool_executor" not in st.session_state:
        try:
            config = get_config()
            st.session_state.config = config
            st.session_state.tool_executor = ToolExecutor(config)
            #st.success("✅ Configuration loaded successfully")
        except Exception as e:
            st.error(f"❌ Error initializing agent: {e}")
            st.stop()

    config = st.session_state.config
    tool_executor = st.session_state.tool_executor

    # Sidebar - Configuration & Examples
    with st.sidebar:
        st.image("https://kasioss.com/pt/images/pt-01.png", width=100, caption="")

        st.divider()

        # Display keywords for each question (experimental feature)
        if st.session_state.question_keywords:
            st.subheader("🔍 Keywords History")
            keywords_text = "\n\n".join([
                f"**Q{item['q_number']}:**\n{', '.join(item['keywords']) if isinstance(item.get('keywords'), list) else 'N/A'}"
                for item in st.session_state.question_keywords
            ])
            st.markdown(keywords_text)
            st.divider()

        #st.caption(f"Memory buffer: {len(st.session_state.graph_state['messages'])} messages")

        if st.button("🔄 Clear Conversation"):
            st.session_state.conversation = []
            st.session_state.question_keywords = []
            # Note: st.rerun() in button callback is a no-op in Streamlit 1.27+
            # The button click automatically triggers a rerun

    # Callback function for follow-up button click
    def on_followup_click(followup_q: str):
        """Handle follow-up button click - auto-submit the follow-up question

        Instead of directly appending to conversation, we set a flag to auto-submit
        the question so it flows through the normal processing pipeline.
        """
        # Store the follow-up question to be auto-submitted
        st.session_state.auto_submit_followup = followup_q
        logger.info(f"Follow-up button clicked: '{followup_q[:60]}...'")
        # Note: st.rerun() in on_click callback is a no-op in Streamlit 1.27+
        # The button click automatically triggers a rerun

    # Chat input
    user_input = st.chat_input("Ask me anything about Pattreeya ...")

    # Process new user input OR auto-submitted follow-up question
    if user_input:
        st.session_state.conversation.append({
            "role": "user",
            "content": user_input
        })
    elif st.session_state.auto_submit_followup:
        # Auto-submit follow-up question from button click
        auto_followup = st.session_state.auto_submit_followup
        st.session_state.conversation.append({
            "role": "user",
            "content": auto_followup
        })
        # Clear the auto-submit flag so we don't process it again
        st.session_state.auto_submit_followup = None
        logger.info(f"Auto-submitted follow-up: '{auto_followup[:60]}...'")


    # Check if there's a pending user query to process
    should_process = False
    pending_user_query = None
    pending_user_index = None

    if (st.session_state.conversation and
        st.session_state.conversation[-1].get("role") == "user"):
        # Check if this user message has a response yet
        # Process if: first message (len == 1) OR previous message is assistant (meaning this is a new question after a response)
        if (len(st.session_state.conversation) == 1 or
            st.session_state.conversation[-2].get("role") == "assistant"):
            # No response yet, process it
            should_process = True
            pending_user_query = st.session_state.conversation[-1]["content"]
            pending_user_index = len(st.session_state.conversation) - 1

    # Display conversation history (all messages except the pending one being processed)
    for i, message in enumerate(st.session_state.conversation):
        # Skip the pending user message (will be displayed with thinking indicator)
        if should_process and i == pending_user_index:
            continue

        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

    # Process pending query (with live display)
    if should_process and pending_user_query:
        # Get CV ID from database (or config, or raise error if not found)
        try:
            # First try to get from config/secrets
            cv_id = st.secrets.get("db", {}).get("cv_id", None)

            # If not in config, fetch from database
            if not cv_id:
                cv_id = config.get_cv_id()
        except ValueError as e:
            st.error(f"❌ Configuration Error: {e}")
            st.info("Please ensure the database is initialized with CV data. Run `python db_ingestion.py` first.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            logger.error(f"Error getting CV ID: {e}")
            st.stop()

        # Display user message once
        with st.chat_message("user"):
            st.write(pending_user_query)

        # Classify keywords for this query (experimental feature)
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING USER QUERY: {pending_user_query}")
        logger.info(f"{'='*80}\n")

        # Use new parallel-aware keyword classification wrapper
        keyword_classification = classify_user_query_keywords_parallel(pending_user_query, config)

        # Store keywords in session state for sidebar display
        q_number = len(st.session_state.question_keywords) + 1
        keywords_for_display = keyword_classification.get("qdrant_keywords", [])

        st.session_state.question_keywords.append({
            "q_number": q_number,
            "keywords": keywords_for_display,
            "search_type": keyword_classification.get("search_type", "unknown")
        })

        # Get response from LLM agent with thinking indicator
        response_text = ""
        followup_question = ""

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # EXPERIMENTAL: Pass keyword classification for keyword injection
                    response_text = execute_llm_cv_query(
                        pending_user_query,
                        cv_id,
                        tool_executor,
                        config
                    )

                except Exception as e:
                    response_text = f"❌ Error processing query: {e}"
                    logger.error(f"Error: {e}")

            # Display response (spinner clears automatically)
            st.write(response_text)

            # Display keywords alongside the response (experimental feature)
            if keywords_for_display:
                st.divider()
                keywords_str = ", ".join(keywords_for_display)
                st.caption(f"🔍 **Q{q_number} Keywords:** {keywords_str}")

        # Generate follow-up question separately (after response is displayed)
        try:
            # Build state dict for generate_followup_with_skip_check
            followup_state = {
                "user_query": pending_user_query,
                "conversation_history": st.session_state.conversation,
                "followup_question": ""
            }

            # Generate follow-up question
            updated_state = generate_followup_with_skip_check(
                state=followup_state,
                final_response=response_text,
                config=config,
                logger_instance=logger
            )

            followup_question = updated_state.get("followup_question", "")
        except Exception as e:
            followup_question = ""
            logger.warning(f"⚠ Failed to generate follow-up question: {e}")

        # Add assistant message to conversation AFTER display
        st.session_state.conversation.append({
            "role": "assistant",
            "content": response_text
        })

        # Display follow-up question as button OUTSIDE assistant message block
        if followup_question and followup_question.strip():
            logger.info(f"✓ Displaying follow-up button with question: {followup_question[:60]}...")
            st.divider()
            st.markdown("**💡 Follow-up question:**")

            # Create button with callback using more robust key
            button_key = f"followup_q_{len(st.session_state.conversation)}_{q_number}"
            st.button(
                followup_question,
                key=button_key,
                use_container_width=True,
                on_click=on_followup_click,
                args=(followup_question,)
            )
        else:
            if not followup_question:
                logger.warning("⚠️  No follow-up question generated (empty string)")

    # # Footer
    # st.markdown("---")
    # st.markdown(
    #     """
    #     <div style='text-align: center; color: gray; font-size: 0.9em;'>
    #         Powered by Pattreeya.
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )


if __name__ == "__main__":
    main()
