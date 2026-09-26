from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import stat
import threading
from unittest.mock import patch

import pytest

from reflectlog.core.enums import EmbedderProvider
from reflectlog.core.exceptions import InitializationError
from reflectlog.core.storage_coordination import IStorageCoordinator, LeaseMode
from reflectlog.core.types import Embeddings
from reflectlog.infrastructure.embedding_identity import (
    IDENTITY_NAME,
    ensure_embedding_identity,
    preflight_embedding_identity,
)
from reflectlog.infrastructure.storage_coordinator import PortalockerStorageCoordinator
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine


class CountingEmbedder(Embeddings):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


@pytest.fixture
def config(tmp_path: Path) -> USearchConfig:
    return USearchConfig(
        workspace_id="alpha",
        index_path=str(tmp_path / "alpha" / "usearch" / "vectors.usearch"),
        db_path=str(tmp_path / "alpha" / "usearch" / "memories.db"),
        embedder_provider=EmbedderProvider.OPENAI,
        embedding_model="test/model",
        embedding_dims=2,
    )


def create_empty_known_sqlite(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE TABLE memories (id INTEGER, workspace_id TEXT, content TEXT, created_at TEXT)"
        )
        connection.execute(
            "CREATE TABLE archived_memories (id INTEGER, original_id INTEGER, "
            "workspace_id TEXT, content TEXT, replaced_by TEXT, reason TEXT, "
            "confidence REAL, archived_at TEXT)"
        )
        connection.execute(
            "CREATE TABLE replacement_transitions (id INTEGER, workspace_id TEXT, "
            "old_memory_id INTEGER, old_content TEXT, new_content TEXT, archive_id INTEGER, "
            "reason TEXT, confidence REAL, status TEXT, created_at TEXT, updated_at TEXT, "
            "kind TEXT)"
        )
        connection.commit()


def test_preflight_is_read_only_when_workspace_is_absent(config: USearchConfig) -> None:
    root = Path(config.index_path).parent.parent

    assert preflight_embedding_identity(config) is False
    assert not root.exists()


def test_first_open_publishes_identity_in_workspace_root(config: USearchConfig) -> None:
    embedder = CountingEmbedder()
    engine = USearchEngine(config=config, embedder=embedder)
    root = Path(config.index_path).parent.parent

    assert json.loads((root / IDENTITY_NAME).read_text()) == {
        "version": 1,
        "workspace_id": "alpha",
        "provider": "openai",
        "model": "test/model",
        "dimensions": 2,
    }
    assert embedder.calls == 0
    engine.close()


@pytest.mark.parametrize(
    "change",
    [
        {"embedder_provider": EmbedderProvider.WEMM},
        {"embedding_model": "test/another"},
        {"embedding_dims": 3},
        {"workspace_id": "beta"},
    ],
)
def test_direct_open_rejects_mismatch_before_inference(
    config: USearchConfig, change: dict[str, str | int | EmbedderProvider]
) -> None:
    ensure_embedding_identity(config)
    root = Path(config.index_path).parent.parent
    before = (root / IDENTITY_NAME).read_bytes()
    embedder = CountingEmbedder()
    mismatched = replace(config, **change)

    with pytest.raises(InitializationError):
        USearchEngine(config=mismatched, embedder=embedder)

    assert embedder.calls == 0
    assert (root / IDENTITY_NAME).read_bytes() == before


def test_dict_config_cannot_bypass_identity(config: USearchConfig) -> None:
    ensure_embedding_identity(config)
    data = {
        "workspace_id": config.workspace_id,
        "index_path": config.index_path,
        "db_path": config.db_path,
        "embedding_dims": config.embedding_dims,
        "embedder_provider": "openai",
        "embedding_model": "test/other",
    }

    with pytest.raises(InitializationError, match="does not match"):
        USearchEngine(config=data, embedder=CountingEmbedder())


@pytest.mark.parametrize(
    "artifact",
    [
        "vectors.usearch",
        "tantivy/metadata.json",
        "memories.db-wal",
        "usearch/recovery.tmp",
    ],
)
def test_legacy_artifacts_refuse_identity_publication(
    config: USearchConfig, artifact: str
) -> None:
    root = Path(config.index_path).parent.parent
    path = root / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"legacy")

    with pytest.raises(InitializationError, match="Legacy workspace"):
        ensure_embedding_identity(config)

    assert not (root / IDENTITY_NAME).exists()


@pytest.mark.parametrize("directory", ["tantivy", "unexpected", "usearch/recovery"])
def test_unknown_empty_legacy_directory_is_rejected(
    config: USearchConfig, directory: str
) -> None:
    root = Path(config.index_path).parent.parent
    (root / directory).mkdir(parents=True)

    with pytest.raises(InitializationError, match="Legacy workspace") as error:
        ensure_embedding_identity(config)

    assert "export" in str(error.value).lower()
    assert "offline" in str(error.value).lower()
    assert not (root / IDENTITY_NAME).exists()


def test_empty_configured_usearch_directory_is_allowed(config: USearchConfig) -> None:
    Path(config.index_path).parent.mkdir(parents=True)

    assert preflight_embedding_identity(config) is False


def test_external_tantivy_index_requires_known_empty_directory(
    config: USearchConfig, tmp_path: Path
) -> None:
    external = tmp_path / "external" / "tantivy"
    external.mkdir(parents=True)
    assert (
        preflight_embedding_identity(config, tantivy_index_path=str(external)) is False
    )
    (external / "metadata.json").write_text("legacy")

    with pytest.raises(InitializationError, match="Legacy workspace"):
        ensure_embedding_identity(config, tantivy_index_path=str(external))

    assert not (tmp_path / "alpha").exists()


def test_external_tantivy_index_unreadable_fails_closed(
    config: USearchConfig, tmp_path: Path
) -> None:
    external = tmp_path / "external" / "tantivy"
    external.mkdir(parents=True)

    with (
        patch(
            "reflectlog.infrastructure.embedding_identity.os.scandir",
            side_effect=PermissionError("denied"),
        ),
        pytest.raises(InitializationError, match="Legacy workspace"),
    ):
        ensure_embedding_identity(config, tantivy_index_path=str(external))

    assert not (tmp_path / "alpha").exists()


@pytest.mark.parametrize(
    "table",
    ["memories", "archived_memories", "replacement_transitions", "unknown_table"],
)
def test_legacy_sqlite_rows_or_unknown_tables_are_rejected(
    config: USearchConfig, table: str
) -> None:
    path = Path(config.db_path)
    create_empty_known_sqlite(path)
    with closing(sqlite3.connect(path)) as connection:
        if table == "unknown_table":
            connection.execute('CREATE TABLE "unknown_table" (content TEXT)')
        elif table == "memories":
            connection.execute(
                'INSERT INTO memories (workspace_id, content) VALUES ("alpha", "old")'
            )
        elif table == "archived_memories":
            connection.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, confidence) "
                'VALUES (1, "alpha", "old", "new", "replaced", 1.0)'
            )
        else:
            connection.execute(
                "INSERT INTO replacement_transitions "
                "(workspace_id, old_memory_id, old_content, new_content, archive_id, "
                "reason, confidence, status) "
                'VALUES ("alpha", 1, "old", "new", 1, "replaced", 1.0, "pending")'
            )
        connection.commit()

    with pytest.raises(InitializationError, match="Legacy workspace"):
        ensure_embedding_identity(config)

    assert not (path.parent.parent / IDENTITY_NAME).exists()


def test_empty_known_sqlite_tables_allow_publication(config: USearchConfig) -> None:
    path = Path(config.db_path)
    create_empty_known_sqlite(path)

    ensure_embedding_identity(config)

    assert preflight_embedding_identity(config)


def test_preflight_does_not_modify_empty_sqlite(config: USearchConfig) -> None:
    path = Path(config.db_path)
    create_empty_known_sqlite(path)
    before = path.read_bytes()

    assert preflight_embedding_identity(config) is False
    assert path.read_bytes() == before
    assert not (path.parent.parent / IDENTITY_NAME).exists()


def test_unknown_schema_on_known_table_is_rejected(config: USearchConfig) -> None:
    path = Path(config.db_path)
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE memories (content TEXT)")

    with pytest.raises(InitializationError, match="Legacy workspace"):
        ensure_embedding_identity(config)

    assert not (path.parent.parent / IDENTITY_NAME).exists()


def test_unknown_index_schema_is_rejected(config: USearchConfig) -> None:
    path = Path(config.db_path)
    create_empty_known_sqlite(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE INDEX unfamiliar ON memories(content)")

    with pytest.raises(InitializationError, match="Legacy workspace"):
        ensure_embedding_identity(config)

    assert not (path.parent.parent / IDENTITY_NAME).exists()


def test_symlink_and_unreadable_sqlite_fail_closed(config: USearchConfig) -> None:
    path = Path(config.db_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not sqlite")

    with pytest.raises(InitializationError):
        ensure_embedding_identity(config)
    path.unlink()
    path.symlink_to(path.parent / "missing")
    with pytest.raises(InitializationError):
        ensure_embedding_identity(config)


def test_malformed_metadata_does_not_get_overwritten(config: USearchConfig) -> None:
    root = Path(config.index_path).parent.parent
    root.mkdir(parents=True)
    sidecar = root / IDENTITY_NAME
    sidecar.write_text('{"version": 2}')

    with pytest.raises(InitializationError):
        ensure_embedding_identity(config)

    assert sidecar.read_text() == '{"version": 2}'


@pytest.mark.parametrize(
    "metadata",
    [
        '{"version": 2}',
        '{"version": 1, "workspace_id": "beta", '
        '"provider": "openai", "model": "test/model", "dimensions": 2}',
    ],
)
def test_incompatible_metadata_provides_offline_recovery_instructions(
    config: USearchConfig, metadata: str
) -> None:
    root = Path(config.index_path).parent.parent
    root.mkdir(parents=True)
    (root / IDENTITY_NAME).write_text(metadata)

    with pytest.raises(InitializationError) as error:
        preflight_embedding_identity(config)

    assert "export" in str(error.value).lower()
    assert "rebuild" in str(error.value).lower()
    assert "offline" in str(error.value).lower()


def test_unreadable_metadata_provides_offline_recovery_instructions(
    config: USearchConfig,
) -> None:
    root = Path(config.index_path).parent.parent
    root.mkdir(parents=True)
    (root / IDENTITY_NAME).write_bytes(b"\xff")

    with pytest.raises(InitializationError) as error:
        preflight_embedding_identity(config)

    assert "export" in str(error.value).lower()
    assert "rebuild" in str(error.value).lower()
    assert "offline" in str(error.value).lower()


def test_failed_replace_cleans_temp_and_does_not_publish(config: USearchConfig) -> None:
    root = Path(config.index_path).parent.parent
    with patch(
        "reflectlog.infrastructure.embedding_identity.os.replace",
        side_effect=OSError("disk"),
    ):
        with pytest.raises(OSError, match="disk"):
            ensure_embedding_identity(config)

    assert sorted(path.name for path in root.iterdir()) == [".reflectlog.writer.lock"]


def test_orphaned_identity_temp_is_rejected_after_lease(config: USearchConfig) -> None:
    root = Path(config.index_path).parent.parent
    root.mkdir(parents=True)
    (root / f"{IDENTITY_NAME}.orphan.tmp").write_text("unfinished")

    with pytest.raises(InitializationError, match="Legacy workspace"):
        ensure_embedding_identity(config)

    assert not (root / IDENTITY_NAME).exists()


def test_failed_file_fsync_cleans_temp_without_publication(
    config: USearchConfig,
) -> None:
    root = Path(config.index_path).parent.parent
    with patch(
        "reflectlog.infrastructure.embedding_identity.os.fsync",
        side_effect=OSError("disk"),
    ):
        with pytest.raises(OSError, match="disk"):
            ensure_embedding_identity(config)

    assert sorted(path.name for path in root.iterdir()) == [".reflectlog.writer.lock"]


def test_directory_fsync_failure_leaves_atomic_identity_for_retry(
    config: USearchConfig,
) -> None:
    if os.name == "nt":
        pytest.skip("Windows does not support syncing directory descriptors")
    root = Path(config.index_path).parent.parent
    sidecar = root / IDENTITY_NAME
    real_fsync = os.fsync

    def fail_directory_sync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("disk")
        real_fsync(descriptor)

    with patch(
        "reflectlog.infrastructure.embedding_identity.os.fsync",
        side_effect=fail_directory_sync,
    ):
        with pytest.raises(OSError, match="disk"):
            ensure_embedding_identity(config)

    before = sidecar.read_bytes()
    with patch(
        "reflectlog.infrastructure.embedding_identity.os.fsync", wraps=real_fsync
    ) as sync:
        ensure_embedding_identity(config)

    assert sidecar.read_bytes() == before
    assert sync.call_count == 1


def test_matching_open_waits_for_first_publication_to_sync(
    config: USearchConfig,
) -> None:
    replaced = threading.Event()
    release = threading.Event()
    second_preflight = threading.Event()
    real_replace = os.replace
    real_preflight = preflight_embedding_identity

    def pause_after_replace(source: str, destination: str | os.PathLike[str]) -> None:
        real_replace(source, destination)
        replaced.set()
        if not release.wait(timeout=5):
            raise TimeoutError("first publisher was not released")

    def observe_preflight(
        candidate: USearchConfig,
        coordinator: IStorageCoordinator | None = None,
        *,
        tantivy_index_path: str | None = None,
    ) -> bool:
        result = real_preflight(
            candidate, coordinator, tantivy_index_path=tantivy_index_path
        )
        if result:
            second_preflight.set()
        return result

    with (
        patch(
            "reflectlog.infrastructure.embedding_identity.os.replace",
            side_effect=pause_after_replace,
        ),
        patch(
            "reflectlog.infrastructure.embedding_identity.preflight_embedding_identity",
            side_effect=observe_preflight,
        ),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        first = executor.submit(ensure_embedding_identity, config)
        try:
            assert replaced.wait(timeout=5)
            second = executor.submit(ensure_embedding_identity, config)
            assert second_preflight.wait(timeout=5)
            assert not second.done()
        finally:
            release.set()
        assert first.result(timeout=5)
        assert second.result(timeout=5)


def test_concurrent_first_opens_publish_once(config: USearchConfig) -> None:
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(ensure_embedding_identity, [config] * 8))

    assert len(results) == 8
    assert preflight_embedding_identity(config)
    assert (
        len(list(Path(config.index_path).parent.parent.glob(f"{IDENTITY_NAME}*"))) == 1
    )


def test_concurrent_incompatible_opens_have_single_winner(
    config: USearchConfig,
) -> None:
    other = replace(config, embedding_model="different")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(ensure_embedding_identity, item) for item in (config, other)
        ]
        outcomes = [future.exception() for future in futures]

    assert sum(result is None for result in outcomes) == 1
    assert sum(isinstance(result, InitializationError) for result in outcomes) == 1
    winner = config if outcomes[0] is None else other
    assert preflight_embedding_identity(winner)


def test_shared_lease_cannot_upgrade_for_publication(
    config: USearchConfig, tmp_path: Path
) -> None:
    coordinator = PortalockerStorageCoordinator(str(tmp_path))
    with coordinator.acquire("alpha", LeaseMode.SHARED):
        with pytest.raises(InitializationError, match="shared lease"):
            ensure_embedding_identity(config, coordinator)

    assert not (tmp_path / "alpha" / IDENTITY_NAME).exists()


def test_matching_open_cannot_upgrade_shared_lease(
    config: USearchConfig, tmp_path: Path
) -> None:
    coordinator = PortalockerStorageCoordinator(str(tmp_path))
    ensure_embedding_identity(config, coordinator)
    with coordinator.acquire("alpha", LeaseMode.SHARED):
        with pytest.raises(InitializationError, match="shared lease"):
            ensure_embedding_identity(config, coordinator)


def test_coordinator_must_point_to_same_workspace(
    config: USearchConfig, tmp_path: Path
) -> None:
    coordinator = PortalockerStorageCoordinator(str(tmp_path / "different"))

    with pytest.raises(InitializationError, match="coordinator root"):
        ensure_embedding_identity(config, coordinator)

    assert not (tmp_path / "alpha").exists()


def test_rejection_preserves_generation(config: USearchConfig, tmp_path: Path) -> None:
    coordinator = PortalockerStorageCoordinator(str(tmp_path))
    ensure_embedding_identity(config, coordinator)
    coordinator.publish_generation("alpha", 7)

    with pytest.raises(InitializationError):
        ensure_embedding_identity(replace(config, embedding_model="other"), coordinator)

    assert coordinator.read_generation("alpha") == 7
