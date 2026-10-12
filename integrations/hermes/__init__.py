"""ReflectLog memory provider; dependencies load only when Hermes registers it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .provider import ReflectLogMemoryProvider


class RegistrationContext(Protocol):
    def register_memory_provider(self, provider: ReflectLogMemoryProvider) -> None: ...


def register(ctx: RegistrationContext) -> None:
    from .provider import ReflectLogMemoryProvider

    ctx.register_memory_provider(ReflectLogMemoryProvider())
