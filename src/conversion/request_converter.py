import json
from typing import Dict, Any, List
from venv import logger
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest, ClaudeMessage
from src.core.config import config
import logging

logger = logging.getLogger(__name__)


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest, model_manager
) -> tuple[Dict[str, Any], Dict[str, str]]:
    """Convert Claude API request format to OpenAI format.

    Returns:
        tuple: (openai_request, tool_name_mapping)
        - openai_request: The converted OpenAI format request
        - tool_name_mapping: Dict mapping sanitized tool names back to original names
    """

    # Map model
    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)

    # Initialize tool name mapping for reverse lookups
    tool_name_mapping = {}

    # Convert messages
    openai_messages = []

    # Add system message if present
    if claude_request.system:
        system_text = ""
        if isinstance(claude_request.system, str):
            system_text = claude_request.system
        elif isinstance(claude_request.system, list):
            text_parts = []
            for block in claude_request.system:
                if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                    text_parts.append(block.text)
                elif (
                    isinstance(block, dict)
                    and block.get("type") == Constants.CONTENT_TEXT
                ):
                    text_parts.append(block.get("text", ""))
            system_text = "\n\n".join(text_parts)

        if system_text.strip():
            openai_messages.append(
                {"role": Constants.ROLE_SYSTEM, "content": system_text.strip()}
            )

    # Process Claude messages
    i = 0
    while i < len(claude_request.messages):
        msg = claude_request.messages[i]

        if msg.role == Constants.ROLE_USER:
            openai_message = convert_claude_user_message(msg)
            openai_messages.append(openai_message)
        elif msg.role == Constants.ROLE_ASSISTANT:
            openai_message = convert_claude_assistant_message(msg)
            openai_messages.append(openai_message)

            # Check if next message contains tool results
            if i + 1 < len(claude_request.messages):
                next_msg = claude_request.messages[i + 1]
                if (
                    next_msg.role == Constants.ROLE_USER
                    and isinstance(next_msg.content, list)
                    and any(
                        block.type == Constants.CONTENT_TOOL_RESULT
                        for block in next_msg.content
                        if hasattr(block, "type")
                    )
                ):
                    # Process tool results
                    i += 1  # Skip to tool result message
                    tool_results = convert_claude_tool_results(next_msg)
                    openai_messages.extend(tool_results)

        i += 1

    # Build OpenAI request
    openai_request = {
        "model": openai_model,
        "messages": openai_messages,
        "max_tokens": min(
            max(claude_request.max_tokens, config.min_tokens_limit),
            config.max_tokens_limit,
        ),
        "temperature": claude_request.temperature,
        "stream": claude_request.stream,
    }
    logger.debug(
        f"Converted Claude request to OpenAI format: {json.dumps(openai_request, indent=2, ensure_ascii=False)}"
    )
    # Add optional parameters
    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None:
        openai_request["top_p"] = claude_request.top_p

    # Convert tools based on configuration
    if claude_request.tools:
        if config.tooling_api == "kosong":
            # Use Kosong/Kimi tooling format
            kimi_tools, kimi_mapping = convert_tools_to_kimi_format_with_mapping(claude_request.tools[:config.max_tools_limit])
            if kimi_tools:
                openai_request["tools"] = kimi_tools
                tool_name_mapping.update(kimi_mapping)
        else:
            # Use standard OpenAI tooling format
            openai_tools = convert_tools_to_openai_format(claude_request.tools[:config.max_tools_limit])
            if openai_tools:
                openai_request["tools"] = openai_tools
                # For OpenAI format, no sanitization needed, so mapping is 1:1
                for tool in claude_request.tools[:config.max_tools_limit]:
                    if tool.name and tool.name.strip():
                        tool_name_mapping[tool.name] = tool.name

    # Convert tool choice
    if claude_request.tool_choice:
        choice_type = claude_request.tool_choice.get("type")
        if choice_type == "auto":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "any":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "tool" and "name" in claude_request.tool_choice:
            openai_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": claude_request.tool_choice["name"]},
            }
        else:
            openai_request["tool_choice"] = "auto"

    return openai_request, tool_name_mapping


def convert_claude_user_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude user message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_USER, "content": ""}
    
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_USER, "content": msg.content}

    # Handle multimodal content
    openai_content = []
    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            openai_content.append({"type": "text", "text": block.text})
        elif block.type == Constants.CONTENT_IMAGE:
            # Convert Claude image format to OpenAI format
            if (
                isinstance(block.source, dict)
                and block.source.get("type") == "base64"
                and "media_type" in block.source
                and "data" in block.source
            ):
                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                        },
                    }
                )

    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": Constants.ROLE_USER, "content": openai_content[0]["text"]}
    else:
        return {"role": Constants.ROLE_USER, "content": openai_content}


def convert_claude_assistant_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format."""
    text_parts = []
    tool_calls = []

    if msg.content is None:
        return {"role": Constants.ROLE_ASSISTANT, "content": None}
    
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_ASSISTANT, "content": msg.content}

    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            text_parts.append(block.text)
        elif block.type == Constants.CONTENT_TOOL_USE:
            # Sanitize tool name if using Kimi tooling API
            tool_name = block.name
            if config.tooling_api == "kosong":
                tool_name = sanitize_tool_name_for_kimi(block.name)

            tool_calls.append(
                {
                    "id": block.id,
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": tool_name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                }
            )

    openai_message = {"role": Constants.ROLE_ASSISTANT}

    # Set content
    if text_parts:
        openai_message["content"] = "".join(text_parts)
    else:
        openai_message["content"] = None

    # Set tool calls
    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return openai_message


def convert_claude_tool_results(msg: ClaudeMessage) -> List[Dict[str, Any]]:
    """Convert Claude tool results to OpenAI format."""
    tool_messages = []

    if isinstance(msg.content, list):
        for block in msg.content:
            if block.type == Constants.CONTENT_TOOL_RESULT:
                content = parse_tool_result_content(block.content)
                tool_messages.append(
                    {
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": block.tool_use_id,
                        "content": content,
                    }
                )

    return tool_messages


def convert_tools_to_openai_format(tools):
    """Convert Claude tools to OpenAI format."""
    openai_tools = []
    for tool in tools:
        if tool.name and tool.name.strip():
            openai_tools.append(
                {
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
            )
    return openai_tools


def sanitize_tool_name_for_kimi(name):
    """Sanitize tool name for Kimi API requirements.

    Kimi requires tool names to:
    - Start with a letter
    - Contain only letters, numbers, underscores, and dashes
    """
    import re

    # Replace double underscores with single ones for MCP tool names
    sanitized = re.sub(r'__+', '_', name)

    # Remove invalid characters, keep only letters, numbers, underscores, and dashes
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', sanitized)

    # Replace multiple consecutive underscores with single ones
    sanitized = re.sub(r'_+', '_', sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    # Ensure it starts with a letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = 'tool_' + sanitized

    # Ensure it's not empty
    if not sanitized:
        sanitized = 'unknown_tool'

    return sanitized


def convert_tools_to_kimi_format_with_mapping(tools):
    """Convert Claude tools to Kimi/Kosong format and return name mapping.

    Returns:
        tuple: (kimi_tools, tool_name_mapping)
        - kimi_tools: List of tools in Kimi format
        - tool_name_mapping: Dict mapping sanitized names back to original names
    """
    kimi_tools = []
    tool_name_mapping = {}

    for tool in tools:
        if tool.name and tool.name.strip():
            logger.debug(f"Processing tool: {tool.name}")

            # Check if this is a Kimi builtin function (starts with $)
            if tool.name.startswith("$"):
                kimi_tools.append(
                    {
                        "type": "builtin_function",
                        "function": {
                            "name": tool.name,
                            # Builtin functions don't need description and parameters
                        },
                    }
                )
                # No sanitization for builtin functions
                tool_name_mapping[tool.name] = tool.name
            else:
                # Sanitize tool name for Kimi API compatibility
                sanitized_name = sanitize_tool_name_for_kimi(tool.name)
                logger.debug(f"Sanitized tool name: {tool.name} -> {sanitized_name}")

                # Store mapping for reverse lookup
                tool_name_mapping[sanitized_name] = tool.name

                # Use standard OpenAI format for custom tools
                kimi_tools.append(
                    {
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": sanitized_name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    }
                )

    logger.debug(f"Final kimi_tools: {json.dumps(kimi_tools, indent=2, ensure_ascii=False)}")
    logger.debug(f"Tool name mapping: {tool_name_mapping}")
    return kimi_tools, tool_name_mapping

def convert_tools_to_kimi_format(tools):
    """Convert Claude tools to Kimi/Kosong format. Legacy function for backward compatibility."""
    kimi_tools, _ = convert_tools_to_kimi_format_with_mapping(tools)
    return kimi_tools


def parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == Constants.CONTENT_TEXT:
                result_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    result_parts.append(item.get("text", ""))
                else:
                    try:
                        result_parts.append(json.dumps(item, ensure_ascii=False))
                    except:
                        result_parts.append(str(item))
        return "\n".join(result_parts).strip()

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content, ensure_ascii=False)
        except:
            return str(content)

    try:
        return str(content)
    except:
        return "Unparseable content"
