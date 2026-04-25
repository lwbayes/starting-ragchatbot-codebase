import json
from typing import List, Optional, Dict, Any
from openai import OpenAI


class AIGenerator:
    """Handles interactions with an OpenAI-compatible chat model (via OpenRouter)."""

    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800,
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        """

        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        api_params = {**self.base_params, "messages": messages}

        if tools:
            api_params["tools"] = [self._convert_tool(t) for t in tools]
            api_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**api_params)
        message = response.choices[0].message

        if message.tool_calls and tool_manager:
            return self._handle_tool_execution(message, api_params, tool_manager)

        return message.content or ""

    def _handle_tool_execution(self, assistant_message, base_params: Dict[str, Any], tool_manager) -> str:
        """
        Execute tool calls returned by the model and request a follow-up completion.
        """
        messages = list(base_params["messages"])

        messages.append({
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_message.tool_calls
            ],
        })

        for tc in assistant_message.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            tool_result = tool_manager.execute_tool(tc.function.name, **args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

        final_params = {**self.base_params, "messages": messages}
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content or ""

    @staticmethod
    def _convert_tool(anthropic_tool: Dict[str, Any]) -> Dict[str, Any]:
        """Translate an Anthropic-style tool definition to OpenAI's function-tool shape."""
        return {
            "type": "function",
            "function": {
                "name": anthropic_tool["name"],
                "description": anthropic_tool.get("description", ""),
                "parameters": anthropic_tool.get(
                    "input_schema",
                    {"type": "object", "properties": {}},
                ),
            },
        }
