"""
Agentic Thinking Toolkit — special token markers, tool registry, planning scaffold.
Uses Qwen3 chat format (<|im_start|>/<|im_end|>) — no custom tokenizer needed.

Format conventions:
  <|im_start|>user\n{query}<|im_end|>
  <|im_start|>assistant\n<think>{reasoning}</think>\n<tool>{call}</tool>\n{answer}<|im_end|>
"""
import json
import re
from mesh_tokenizer import load_tokenizer


# ── Domain markers (prefixes for router domain detection) ──
DOMAIN_PREFIXES = {
    "reasoning": "<think>",
    "tool_use": "<tool>",
    "planning": "<plan>",
    "general": "<|im_start|>",
}


# ── Tool Registry ──

class Tool:
    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON schema

    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


TOOL_REGISTRY: dict[str, Tool] = {}


def register_tool(name: str, description: str, parameters: dict):
    tool = Tool(name, description, parameters)
    TOOL_REGISTRY[name] = tool
    return tool


def get_tool(name: str) -> Tool | None:
    return TOOL_REGISTRY.get(name)


def list_tools() -> list[dict]:
    return [t.to_schema() for t in TOOL_REGISTRY.values()]


# Pre-register common tools
register_tool(
    "search_web",
    "Search the web for current information",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    },
)

register_tool(
    "run_code",
    "Execute Python code and return output",
    {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
        },
        "required": ["code"],
    },
)

register_tool(
    "read_file",
    "Read a file from disk",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
        },
        "required": ["path"],
    },
)


# ── Format helpers ──

def format_chat(messages: list[dict]) -> str:
    """Convert OpenAI-style messages to Qwen3 chat format."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def format_thinking(thoughts: str, tools: list[str] | None = None) -> str:
    """Wrap reasoning and optional tool calls in thinking markers."""
    parts = ["<think>", thoughts, "</think>"]
    if tools:
        for tool_call in tools:
            parts.append(f"\n<tool>\n{tool_call}\n</tool>")
    return "\n".join(parts)


def format_plan(steps: list[str]) -> str:
    """Format a multi-step plan."""
    plan = "<plan>\n" + "\n".join(f"  - {s}" for s in steps) + "\n</plan>"
    return plan


def parse_thinking(response: str) -> dict:
    """Extract think/tool/answer sections from a model response."""
    think = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    tool = re.search(r"<tool>(.*?)</tool>", response, re.DOTALL)
    plan = re.search(r"<plan>(.*?)</plan>", response, re.DOTALL)
    answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    answer = re.sub(r"<tool>.*?</tool>", "", answer, flags=re.DOTALL)
    answer = re.sub(r"<plan>.*?</plan>", "", answer, flags=re.DOTALL)
    answer = answer.strip().removeprefix("<|im_start|>assistant").strip()
    return {
        "thinking": think.group(1).strip() if think else "",
        "tool_call": tool.group(1).strip() if tool else "",
        "plan": plan.group(1).strip() if plan else "",
        "answer": answer.strip(),
    }


# ── Dataset formatters ──

def format_reasoning(question: str, answer: str, reasoning: str | None = None) -> str:
    """Format a reasoning QA pair with chain-of-thought."""
    if reasoning:
        response = format_thinking(reasoning) + "\n" + answer
    else:
        response = answer
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    return format_chat(messages)


def format_tool_use(query: str, tool_name: str, tool_args: str, result: str) -> str:
    """Format a tool-use interaction."""
    messages = [
        {"role": "user", "content": query},
        {
            "role": "assistant",
            "content": format_thinking(
                f"I need to use {tool_name} to answer this.",
                tools=[f'{tool_name}({tool_args})'],
            ) + "\n" + result,
        },
    ]
    return format_chat(messages)


# ── Domain detection for router ──

def detect_domain(text: str) -> str:
    """Detect domain from text content for router routing."""
    if "<think>" in text:
        return "reasoning"
    if "<tool>" in text:
        return "tool_use"
    if "<plan>" in text:
        return "planning"
    return "general"
