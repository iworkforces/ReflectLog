from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, Required, TypedDict

@dataclass
class TextBlock:
    text: str

@dataclass
class ThinkingBlock:
    thinking: str
    signature: str

@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]

@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None

@dataclass
class ServerToolUseBlock:
    id: str
    name: Literal[
        "advisor",
        "web_search",
        "web_fetch",
        "code_execution",
        "bash_code_execution",
        "text_editor_code_execution",
        "tool_search_tool_regex",
        "tool_search_tool_bm25",
    ]
    input: dict[str, Any]

@dataclass
class ServerToolResultBlock:
    tool_use_id: str
    content: dict[str, Any]

type ContentBlock = (
    TextBlock
    | ThinkingBlock
    | ToolUseBlock
    | ToolResultBlock
    | ServerToolUseBlock
    | ServerToolResultBlock
)
type AssistantMessageError = Literal[
    "authentication_failed",
    "billing_error",
    "rate_limit",
    "invalid_request",
    "server_error",
    "unknown",
]

MessageOrigin = TypedDict(
    "MessageOrigin",
    {
        "kind": Required[
            Literal[
                "human",
                "channel",
                "peer",
                "task-notification",
                "coordinator",
                "unclassified",
                "observer",
                "auto-continuation",
                "observer-activity",
            ]
        ],
        "server": str,
        "from": str,
        "name": str,
        "fromSession": str,
        "senderTaskId": str,
        "body": str,
        "verifiedPeerPid": int,
        "subkind": Literal["scheduled-trigger", "peer-send-message"],
    },
    total=False,
)

@dataclass
class AssistantMessage:
    content: list[ContentBlock]
    model: str
    parent_tool_use_id: str | None = None
    error: AssistantMessageError | None = None
    usage: dict[str, Any] | None = None
    message_id: str | None = None
    stop_reason: str | None = None
    session_id: str | None = None
    uuid: str | None = None

@dataclass
class UserMessage:
    content: str | list[ContentBlock]
    uuid: str | None = None
    parent_tool_use_id: str | None = None
    tool_use_result: dict[str, Any] | None = None
    origin: MessageOrigin | None = None

@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, Any]

class ModelUsage(TypedDict):
    inputTokens: int
    outputTokens: int
    cacheReadInputTokens: int
    cacheCreationInputTokens: int
    webSearchRequests: int
    costUSD: float
    contextWindow: int
    maxOutputTokens: int
    canonicalModel: NotRequired[str]
    provider: NotRequired[str]

@dataclass
class DeferredToolUse:
    id: str
    name: str
    input: dict[str, Any]

@dataclass
class ResultMessage:
    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    stop_reason: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    result: str | None = None
    structured_output: object = None
    model_usage: dict[str, ModelUsage] | None = None
    permission_denials: list[object] | None = None
    deferred_tool_use: DeferredToolUse | None = None
    errors: list[str] | None = None
    api_error_status: int | None = None
    uuid: str | None = None
    terminal_reason: str | None = None
    origin: MessageOrigin | None = None

@dataclass
class StreamEvent:
    uuid: str
    session_id: str
    event: dict[str, Any]
    parent_tool_use_id: str | None = None

@dataclass
class RateLimitInfo:
    status: Literal["allowed", "allowed_warning", "rejected"]
    resets_at: int | None = None
    rate_limit_type: (
        Literal[
            "five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet", "overage"
        ]
        | None
    ) = None
    utilization: float | None = None
    overage_status: Literal["allowed", "allowed_warning", "rejected"] | None = None
    overage_resets_at: int | None = None
    overage_disabled_reason: str | None = None
    raw: dict[str, Any] = ...

@dataclass
class RateLimitEvent:
    rate_limit_info: RateLimitInfo
    uuid: str
    session_id: str

@dataclass
class ConversationResetMessage:
    new_conversation_id: str
    uuid: str
    session_id: str

type Message = (
    UserMessage
    | AssistantMessage
    | SystemMessage
    | ResultMessage
    | StreamEvent
    | RateLimitEvent
    | ConversationResetMessage
)

class ToolsPreset(TypedDict):
    type: Literal["preset"]
    preset: Literal["claude_code"]

class SystemPromptPreset(TypedDict):
    type: Literal["preset"]
    preset: Literal["claude_code"]
    append: NotRequired[str]

type PermissionMode = Literal[
    "default", "acceptEdits", "plan", "bypassPermissions", "dontAsk", "auto"
]

@dataclass
class ClaudeAgentOptions:
    tools: list[str] | ToolsPreset | None = None
    allowed_tools: list[str] = ...
    system_prompt: str | SystemPromptPreset | None = None
    permission_mode: PermissionMode | None = None
    model: str | None = None

def query(
    *, prompt: str, options: ClaudeAgentOptions | None = None
) -> AsyncGenerator[Message]: ...

__all__ = [
    "AssistantMessage",
    "AssistantMessageError",
    "ClaudeAgentOptions",
    "ContentBlock",
    "ConversationResetMessage",
    "DeferredToolUse",
    "Message",
    "MessageOrigin",
    "ModelUsage",
    "PermissionMode",
    "RateLimitEvent",
    "RateLimitInfo",
    "ResultMessage",
    "ServerToolResultBlock",
    "ServerToolUseBlock",
    "StreamEvent",
    "SystemMessage",
    "SystemPromptPreset",
    "TextBlock",
    "ThinkingBlock",
    "ToolResultBlock",
    "ToolUseBlock",
    "ToolsPreset",
    "UserMessage",
    "query",
]
