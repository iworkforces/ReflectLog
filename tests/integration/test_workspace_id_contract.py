"""One workspace rule at real storage boundaries, before filesystem effects."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.config.validation import validate_config
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.memory.workspace_registry import WorkspaceRegistry
from reflectlog.application.utils.logging import create_logger
from reflectlog.core.config_adapters import ConfigAdapter, StorageConfigAdapter
from reflectlog.core.exceptions import ConfigurationError, ValidationError
from reflectlog.core.storage_coordination import LeaseMode
from reflectlog.infrastructure.embedding_identity import (
    IDENTITY_NAME,
    EmbeddingIdentity,
    _DirectCoordinator,
    ensure_embedding_identity,
)
from reflectlog.infrastructure.storage_coordinator import PortalockerStorageCoordinator
from reflectlog.infrastructure.usearch_engine import USearchConfig
from tests.integration.test_memory_manager_usearch import (
    MockEmbedder,
    create_usearch_config,
)

pytestmark = pytest.mark.integration
REJECTED = [
    "",
    " \t\n",
    ".",
    "..",
    "a..b",
    "/abs",
    "a/b",
    "a\\b",
    "a b",
    "a\x00b",
    "a\nb",
    "a\tb",
    "café",
    "Ａ",
    "a" * 65,
    "a" * 128,
    "a" * 129,
]
ACCEPTED = [
    ("a", "a"),
    ("AZ09_.-", "az09_.-"),
    ("A" * 64, "a" * 64),
    (" \tMiXeD\n", "mixed"),
]


@pytest.fixture
def bound_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.chdir(tmp_path)
    return replace(create_usearch_config("indexes"), workspace_id="alpha")


def direct_config(root: Path, workspace_id: str) -> USearchConfig:
    return USearchConfig.from_dict(
        {
            "workspace_id": workspace_id,
            "index_path": str(root / "alpha" / "usearch" / "vectors.usearch"),
            "db_path": str(root / "alpha" / "usearch" / "memories.db"),
            "embedding_dims": 128,
            "embedder_provider": "langchain",
            "embedding_model": "mock-model",
        }
    )


@pytest.mark.parametrize("workspace_id", REJECTED)
@pytest.mark.parametrize(
    "operation", ["paths", "acquire", "held", "read", "publish", "direct"]
)
def test_coordinator_rejects_before_filesystem_effects(
    tmp_path: Path, workspace_id: str, operation: str
) -> None:
    root = tmp_path / "indexes"
    root.mkdir()
    coordinator = PortalockerStorageCoordinator(str(root))
    before = list(root.iterdir())
    with pytest.raises(ValidationError):
        match operation:
            case "paths":
                coordinator.paths_for(workspace_id)
            case "acquire":
                with coordinator.acquire(workspace_id, LeaseMode.EXCLUSIVE):
                    pass
            case "held":
                coordinator.is_held(workspace_id)
            case "read":
                coordinator.read_generation(workspace_id)
            case "publish":
                coordinator.publish_generation(workspace_id, 1)
            case "direct":
                _DirectCoordinator(root / "alpha", workspace_id)
            case _:
                pytest.fail("unknown coordinator operation")
    assert list(root.iterdir()) == before


@pytest.mark.parametrize("workspace_id", REJECTED)
@pytest.mark.parametrize(
    "operation",
    [
        "from_config",
        "from_dict",
        "construct",
        "adapter",
        "storage_adapter",
        "tantivy_adapter",
        "tantivy_storage_adapter",
        "identity",
        "ensure",
    ],
)
def test_config_and_identity_reject_before_filesystem_effects(
    tmp_path: Path, bound_config: Config, workspace_id: str, operation: str
) -> None:
    root = tmp_path / "indexes"
    root.mkdir()
    config = replace(bound_config, workspace_id=workspace_id)
    before = list(root.iterdir())
    with pytest.raises(ValidationError):
        match operation:
            case "from_config":
                USearchConfig.from_config(ConfigAdapter(config))
            case "from_dict":
                direct_config(root, workspace_id)
            case "construct":
                replace(direct_config(root, "alpha"), workspace_id=workspace_id)
            case (
                "adapter"
                | "storage_adapter"
                | "tantivy_adapter"
                | "tantivy_storage_adapter"
            ):
                adapter = (
                    StorageConfigAdapter(config)
                    if "storage" in operation
                    else ConfigAdapter(config)
                )
                if operation.startswith("tantivy"):
                    _ = adapter.tantivy_index_path
                else:
                    _ = adapter.usearch_index_path
            case "identity" | "ensure":
                storage = direct_config(root, "alpha")
                # Exercise the identity boundary independently of USearchConfig's guard.
                object.__setattr__(storage, "workspace_id", workspace_id)
                if operation == "identity":
                    EmbeddingIdentity.from_config(storage)
                else:
                    ensure_embedding_identity(storage)
            case _:
                pytest.fail("unknown storage operation")
    assert list(root.iterdir()) == before


@pytest.mark.parametrize("workspace_id", REJECTED)
def test_direct_manager_rejects_before_filesystem_effects(
    tmp_path: Path, bound_config: Config, workspace_id: str
) -> None:
    root = tmp_path / "indexes"
    root.mkdir()
    before = list(root.iterdir())
    # Application boundaries translate the shared ValidationError to ConfigurationError.
    with pytest.raises(ConfigurationError):
        with closing(
            MemoryManager(
                replace(bound_config, workspace_id=workspace_id),
                create_logger("contract", "test"),
            )
        ):
            pytest.fail("invalid workspace manager constructed")
    assert list(root.iterdir()) == before


@pytest.mark.parametrize("workspace_id", REJECTED)
async def test_registry_rejects_before_filesystem_effects(
    tmp_path: Path, bound_config: Config, workspace_id: str
) -> None:
    root = tmp_path / "indexes"
    root.mkdir()
    before = list(root.iterdir())
    registry = WorkspaceRegistry(bound_config)
    try:
        with pytest.raises(ConfigurationError):
            async with registry.acquire(workspace_id):
                pytest.fail("invalid workspace acquired")
    finally:
        await registry.close()
    assert list(root.iterdir()) == before


@pytest.mark.parametrize("workspace_id,expected", ACCEPTED)
def test_accepted_ids_share_canonical_storage(
    tmp_path: Path, bound_config: Config, workspace_id: str, expected: str
) -> None:
    root = tmp_path / "indexes"
    config = replace(bound_config, workspace_id=workspace_id)
    coordinator = PortalockerStorageCoordinator(str(root))
    storage = USearchConfig.from_config(ConfigAdapter(config))
    assert storage.workspace_id == expected
    assert direct_config(root, workspace_id).workspace_id == expected
    assert replace(storage, workspace_id=workspace_id).workspace_id == expected
    for adapter in (ConfigAdapter(config), StorageConfigAdapter(config)):
        assert adapter.usearch_index_path == f"indexes/{expected}/usearch"
        assert adapter.tantivy_index_path == f"indexes/{expected}/tantivy"
    assert coordinator.paths_for(workspace_id) == coordinator.paths_for(expected)
    ensure_embedding_identity(storage, coordinator)
    assert EmbeddingIdentity.from_config(storage).workspace_id == expected
    assert (root / expected / IDENTITY_NAME).is_file()
    with coordinator.acquire(workspace_id, LeaseMode.EXCLUSIVE):
        assert coordinator.is_held(expected)
        coordinator.publish_generation(workspace_id, 1)
    assert coordinator.read_generation(expected) == 1
    assert [p.name for p in root.iterdir()] == [expected]


@pytest.mark.parametrize("workspace_id,expected", ACCEPTED)
def test_real_manager_normalizes_bound_config_once(
    tmp_path: Path, bound_config: Config, workspace_id: str, expected: str
) -> None:
    config = replace(bound_config, workspace_id=workspace_id)
    with patch(
        "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
        return_value=MockEmbedder(),
    ):
        manager = MemoryManager(config, create_logger("contract", "test"))
    try:
        assert manager.config.workspace_id == manager.workspace_id == expected
        assert config.workspace_id == workspace_id
        assert [p.name for p in (tmp_path / "indexes").iterdir()] == [expected]
        assert (tmp_path / "indexes" / expected / "tantivy").is_dir()
    finally:
        manager.close()


@pytest.mark.parametrize("workspace_id,expected", ACCEPTED)
async def test_registry_aliases_pin_one_real_manager(
    tmp_path: Path, bound_config: Config, workspace_id: str, expected: str
) -> None:
    registry = WorkspaceRegistry(bound_config)
    try:
        with patch(
            "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
            return_value=MockEmbedder(),
        ):
            async with registry.acquire(workspace_id) as first:
                async with registry.acquire(expected) as second:
                    assert first is second
                    assert first.workspace_id == expected
        assert [p.name for p in (tmp_path / "indexes").iterdir()] == [expected]
    finally:
        await registry.close()


@pytest.mark.parametrize("workspace_id", REJECTED)
@pytest.mark.parametrize("allow_unbound", [False, True])
def test_config_template_exception_is_exact_empty_only(
    bound_config: Config, workspace_id: str, allow_unbound: bool
) -> None:
    errors = validate_config(
        replace(bound_config, workspace_id=workspace_id),
        allow_unbound_workspace=allow_unbound,
    )
    workspace_errors = [error for error in errors if error.field == "WORKSPACE_ID"]
    assert bool(workspace_errors) is not (allow_unbound and workspace_id == "")


def test_environment_without_workspace_builds_unbound_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORKSPACE_ID", raising=False)
    monkeypatch.setenv("EMBEDDER_PROVIDER", "wemm")
    monkeypatch.setenv("EMBEDDING_MODEL", "tencent/WeMM-Embedding-2B")
    monkeypatch.setenv("RERANKER_ENGINE", "none")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-admin-password")
    config = Config.from_environment()
    assert config.workspace_id == ""
    assert any(error.field == "WORKSPACE_ID" for error in validate_config(config))
    assert not any(
        error.field == "WORKSPACE_ID"
        for error in validate_config(config, allow_unbound_workspace=True)
    )
