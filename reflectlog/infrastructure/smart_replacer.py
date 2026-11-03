"""Smart memory replacement detector using LLM."""

import asyncio
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.prompts import format_replacement_detection_prompt
from reflectlog.infrastructure.llm_provider_base import (
    BaseOpenAIProvider,
    parse_replacement_decision,
)

if TYPE_CHECKING:
    from reflectlog.core.config import IAppConfig


class ReplacementDecision(BaseModel):
    """Schema for LLM replacement detection response.

    Used with OpenAI Structured Outputs to guarantee valid JSON responses
    from the smart replacer.

    Attributes:
        should_replace: Whether the new memory should replace the old one.
        confidence: Confidence score from 0.0 to 1.0.
        reason: Brief explanation of the decision.
    """

    should_replace: bool = Field(
        ...,
        description="Whether the new memory should replace the old one",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0",
    )
    reason: str = Field(
        ...,
        description="Brief explanation of the decision",
    )


@dataclass(frozen=True)
class SmartReplacerConfig:
    """Configuration for SmartReplacer.

    Attributes:
        api_key: OpenRouter API key for authentication.
        base_url: OpenRouter API base URL.
        model: LLM model identifier (e.g., 'x-ai/grok-4.1-fast').
        threshold: Minimum confidence score to trigger replacement (0.0-1.0).
        enabled: Whether smart replacement is enabled.
        timeout: Per-attempt timeout in seconds for replacement detection.
        max_retries: Maximum number of retry attempts for LLM calls.
        retry_delay: Base delay in seconds for exponential backoff.
        provider: LLM provider ('openai').
    """

    api_key: str
    base_url: str
    model: str
    threshold: float = 0.7
    enabled: bool = True
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    provider: str = "openai"

    @classmethod
    def from_config(cls, config: IAppConfig) -> SmartReplacerConfig:
        """Create SmartReplacerConfig from IAppConfig protocol.

        Args:
            config: Configuration satisfying IAppConfig protocol.

        Returns:
            SmartReplacerConfig instance configured from app settings.
        """
        return cls(
            api_key=config.llm_api_key,
            base_url=config.llm_api_base_url,
            model=config.llm_model,
            threshold=config.smart_replace_threshold,
            enabled=config.enable_smart_replace,
            max_retries=config.smart_replace_max_retries,
            retry_delay=config.smart_replace_retry_delay,
            provider=config.llm_provider,
        )


class IReplacementProvider(Protocol):
    """Protocol for replacement detection providers.

    Defines the interface for LLM providers used in smart memory replacement.
    """

    async def detect_replacement(
        self,
        prompt: str,
        max_retries: int,
        retry_delay: float,
    ) -> tuple[bool, float, str]:
        """Detect if replacement should occur.

        Args:
            prompt: The formatted replacement detection prompt.
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay in seconds for exponential backoff.

        Returns:
            Tuple of (should_replace, confidence, reason):
                - should_replace: Whether replacement is recommended
                - confidence: Confidence score (0.0-1.0)
                - reason: Brief explanation of the decision
        """
        ...


class OpenAIReplacementProvider(BaseOpenAIProvider):
    """OpenAI/OpenRouter-based replacement detection provider.

    Uses AsyncOpenAI client with structured JSON output for reliable parsing.
    Supports fallback to json_object mode for models without structured output.

    Inherits common functionality from BaseOpenAIProvider:
    - AsyncOpenAI client initialization with HTTP/2 support
    - Structured output with fallback to json_object
    - Safe JSON parsing with clamping

    Retry logic is handled by the retry decorator.
    """

    async def _detect_replacement_once(
        self,
        prompt: str,
    ) -> tuple[bool, float, str]:
        """Call LLM with structured output once.

        Args:
            prompt: The formatted replacement detection prompt.

        Returns:
            Tuple of (should_replace, confidence, reason).

        Raises:
            Exception: If LLM call fails.
        """
        result = await self._call_llm_with_structured_output(
            prompt=prompt,
            response_schema=ReplacementDecision,
            max_tokens=150,
        )

        return parse_replacement_decision(result)

    async def detect_replacement(
        self,
        prompt: str,
        max_retries: int,
        retry_delay: float,
    ) -> tuple[bool, float, str]:
        """Detect if replacement should occur using OpenAI API.

        Args:
            prompt: The formatted replacement detection prompt.
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay in seconds for exponential backoff.

        Returns:
            Tuple of (should_replace, confidence, reason).
        """
        last_exception: Exception | None = None
        attempts = max(1, max_retries)
        for attempt in range(1, attempts + 1):
            try:
                return await self._detect_replacement_once(prompt)
            except Exception as exc:
                last_exception = exc
                if self._logger:
                    self._logger.warning(
                        "OpenAI replacement detection failed",
                        extra={
                            "attempt": attempt,
                            "max_retries": attempts,
                            "exception_type": type(exc).__name__,
                        },
                    )
                if attempt < attempts:
                    await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))

        error_type = type(last_exception).__name__ if last_exception else "Unknown"
        return (False, 0.0, f"Error: replacement detection failed ({error_type})")


def create_replacement_provider(
    config: SmartReplacerConfig,
    logger: IStructuredLogger | None = None,
) -> IReplacementProvider:
    """Create a replacement provider based on configuration.

    Factory function that returns the appropriate provider implementation
    based on the provider setting in config.

    Args:
        config: SmartReplacerConfig with provider selection.
        logger: Optional structured logger.

    Returns:
        An IReplacementProvider implementation.

    Raises:
        ValueError: If provider is not supported.
    """
    if config.provider == "openai":
        return OpenAIReplacementProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            timeout=config.timeout,
            logger=logger,
        )
    else:
        raise ValueError(
            f"Unsupported provider: '{config.provider}'. Valid options: 'openai'"
        )


class SmartReplacer(BaseModel):
    """LLM-based smart memory replacement detector.

    This class determines if a new memory should replace an existing one
    by analyzing semantic similarity and contextual updates using an LLM.

    Uses the OpenRouter/OpenAI API with structured JSON output.

    Attributes:
        config: SmartReplacerConfig with API credentials and settings.
        logger: Optional structured logger for debug/info memory details.

    Example:
        >>> config = SmartReplacerConfig.from_app_config(app_config)
        >>> replacer = SmartReplacer(config=config, logger=logger)
        >>> should_replace, confidence, reason = await replacer.check_replacement(
        ...     new_memory="I don't like cats anymore",
        ...     existing_memory="I like cats"
        ... )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: SmartReplacerConfig
    logger: IStructuredLogger | None = None

    _provider: IReplacementProvider | None = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        """Initialize SmartReplacer with appropriate provider."""
        super().__init__(**data)

        if self.config.enabled:
            self._provider = create_replacement_provider(self.config, self.logger)

    async def check_replacement(
        self,
        new_memory: str,
        existing_memory: str,
    ) -> tuple[bool, float, str]:
        """Check if new memory should replace existing memory.

        Uses LLM to analyze whether the new memory semantically replaces
        the existing memory (e.g., updated preference, contradictory statement).

        Args:
            new_memory: The new memory being added.
            existing_memory: An existing memory to compare against.

        Returns:
            Tuple of (should_replace, confidence, reason):
                - should_replace: True if replacement should occur
                - confidence: LLM confidence score (0.0-1.0)
                - reason: Brief explanation of the decision
        """
        if not self.config.enabled:
            return (False, 0.0, "Smart replacement disabled")

        if self._provider is None:
            return (False, 0.0, "Provider not initialized")

        try:
            # Format the replacement detection prompt
            prompt = format_replacement_detection_prompt(
                old_memory=existing_memory,
                new_memory=new_memory,
            )

            # Delegate to provider
            (
                should_replace,
                confidence,
                reason,
            ) = await self._provider.detect_replacement(
                prompt=prompt,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
            )

            # Only trigger replacement if confidence meets threshold
            final_should_replace = (
                should_replace and confidence >= self.config.threshold
            )

            if self.logger:
                self.logger.debug(
                    f"Smart replacer decision: should_replace={should_replace}, "
                    f"confidence={confidence:.2f}, threshold={self.config.threshold}",
                    extra={
                        "should_replace": should_replace,
                        "final_should_replace": final_should_replace,
                        "confidence": confidence,
                        "threshold": self.config.threshold,
                        "reason_length": len(reason),
                        "provider": self.config.provider,
                    },
                )

            return (final_should_replace, confidence, reason)

        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.warning(
                    "Invalid JSON from LLM for replacement detection",
                    extra={
                        "exception_type": type(e).__name__,
                        "provider": self.config.provider,
                    },
                )
            return (False, 0.0, "JSON parse error: replacement detection failed")

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "Smart replacement check failed",
                    extra={
                        "exception_type": type(e).__name__,
                        "provider": self.config.provider,
                    },
                )
            return (False, 0.0, f"Error: replacement check failed ({type(e).__name__})")
