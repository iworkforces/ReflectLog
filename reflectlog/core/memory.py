"""Memory operation protocols for ReflectLog.

This module defines protocols for memory storage and retrieval operations.
These abstractions enable different memory backends-whether based on vector
databases, document stores, or hybrid approaches-while presenting a
consistent interface to the application layer.
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from reflectlog.core.enums import EngineReadiness
    from reflectlog.core.types import MemoryRecord


@runtime_checkable
class IMemoryStore(Protocol):
    """Protocol for memory storage operations.

    This protocol defines the interface for storing and retrieving
    individual memory entries.
    """

    async def add(
        self,
        workspace_id: str,
        content: str,
        metadata: MemoryRecord | None = None,
    ) -> str:
        """Add a memory entry to the store.

        Args:
            workspace_id: Workspace identifier for filtering.
            content: Memory content to store.
            metadata: Optional additional metadata.

        Returns:
            Unique identifier for the stored memory.
        """
        ...

    async def get(
        self,
        workspace_id: str,
        memory_id: str,
    ) -> MemoryRecord | None:
        """Retrieve a memory entry by ID.

        Args:
            workspace_id: Workspace identifier.
            memory_id: Memory identifier.

        Returns:
            Memory entry dict or None if not found.
        """
        ...

    async def get_all(
        self,
        workspace_id: str,
    ) -> list[MemoryRecord]:
        """Retrieve all memories for a workspace.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            List of all memory entries.
        """
        ...

    async def delete(
        self,
        workspace_id: str,
        memory_id: str,
    ) -> bool:
        """Delete a memory entry.

        Args:
            workspace_id: Workspace identifier.
            memory_id: Memory identifier.

        Returns:
            True if deleted, False if not found.
        """
        ...

    async def update(
        self,
        workspace_id: str,
        memory_id: str,
        content: str,
        metadata: MemoryRecord | None = None,
    ) -> bool:
        """Update a memory entry.

        Args:
            workspace_id: Workspace identifier.
            memory_id: Memory identifier.
            content: New memory content.
            metadata: Optional new metadata.

        Returns:
            True if updated, False if not found.
        """
        ...

    async def find_by_content(
        self,
        workspace_id: str,
        content: str,
    ) -> MemoryRecord | None:
        """Find a memory by exact content match.

        Args:
            workspace_id: Workspace identifier.
            content: Memory content to find.

        Returns:
            Memory entry dict or None if not found.
        """
        ...

    async def exists(
        self,
        workspace_id: str,
        content: str,
    ) -> bool:
        """Check if a memory with exact content exists.

        Args:
            workspace_id: Workspace identifier.
            content: Memory content to check.

        Returns:
            True if exists, False otherwise.
        """
        ...

    async def count(
        self,
        workspace_id: str,
    ) -> int:
        """Count memories in a project.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            Number of memories.
        """
        ...

    async def close(self) -> None:
        """Release resources and persist data."""
        ...

    async def commit(self) -> None:
        """Commit pending changes to storage."""
        ...


@runtime_checkable
class IMemoryBackend(IMemoryStore, Protocol):
    """Protocol for memory backend operations.

    This protocol extends IMemoryStore with backend-specific operations
    like vector search and full-text search.
    """

    async def search(
        self,
        workspace_id: str,
        query: str,
        limit: int = 10,
    ) -> list[tuple[str, float, str]]:
        """Search memories by semantic similarity.

        Args:
            workspace_id: Workspace identifier.
            query: Search query.
            limit: Maximum results.

        Returns:
            List of (content, score, memory_id) tuples.
        """
        ...

    async def add_batch(
        self,
        workspace_id: str,
        contents: list[str],
    ) -> list[str]:
        """Add multiple memories efficiently.

        Args:
            workspace_id: Workspace identifier.
            contents: List of memory contents to store.

        Returns:
            List of stored memory content strings (may differ from input due to dedup).
        """
        ...

    async def get_id_by_content(
        self,
        workspace_id: str,
        content: str,
    ) -> int | None:
        """Get internal ID by memory content.

        Args:
            workspace_id: Workspace identifier.
            content: Memory content.

        Returns:
            Internal ID or None if not found.
        """
        ...

    def ensure_initialized(self) -> None:
        """Ensure backend is fully initialized (thread-safe)."""
        ...


@runtime_checkable
class IMemoryManager(Protocol):
    """Protocol for memory management facade.

    This protocol defines the interface that the rest of the application
    uses to interact with the memory system. It coordinates multiple
    backends (semantic search, full-text search) and provides a unified
    interface.
    """

    @property
    def workspace_id(self) -> str:
        """Workspace identifier."""
        ...

    async def add_memories_async(
        self,
        memories: list[str],
        dry_run: bool = False,
    ) -> object:
        """Add memories through the 3-phase pipeline."""
        ...

    def add_memories(self, memories: list[str]) -> int:
        """Synchronous add used by tests and recovery seeding."""
        ...

    def count(self) -> int:
        """Return how many memories exist in this workspace."""
        ...

    def pending_intent_count(self) -> int:
        """Return how many add/delete/replace intents are still pending.

        Raises if the journal cannot be listed so health cannot report
        a healthy zero while leftover rows are unreadable.
        """
        ...

    def search_engine_status(self) -> dict[str, EngineReadiness]:
        """Return public readiness of search engines for health checks."""
        ...

    def get_all(self, limit: int | None = None, offset: int = 0) -> list[str]:
        """Retrieve stored memories from the semantic backend."""
        ...

    async def search(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[str]:
        """Hybrid search."""
        ...

    def delete_by_memory(self, memory: str) -> bool:
        """Delete one memory by exact content."""
        ...

    def delete_memories(self, memories: list[str]) -> list[str]:
        """Delete many memories and return the contents that were removed."""
        ...

    def close(self) -> None:
        """Release resources and persist data."""
        ...
