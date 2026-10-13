"""Offline verifier trust boundaries, exercised with real exported bundles."""

# SIZE_OK: one owned test module is required for this boundary-test workstream.
import ast
import builtins
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from threading import Thread
from typing import Literal

import pytest

from reflectlog.application.salvage.verification import (
    ContentSalvageVerifier,
    result_payload,
)
from reflectlog.core.salvage import (
    BundleCapture,
    CapturedFile,
    IBundleReader,
    VerificationResult,
    VerificationStatus,
)
from reflectlog.infrastructure.salvage_bundle_reader import FileSystemBundleReader
from tests.unit.salvage_support import SECRET, WS, as_object, tree_digest
from tests.unit.salvage_verify_support import (
    export_bundle,
    load_manifest,
    record_line,
    store_manifest,
    write_records,
)

REPO = Path(__file__).resolve().parents[4]
type IssueTriple = tuple[str, str, int | None]
type BundleState = Literal["ready", "not-ready", "corrupt", "missing-newline"]
FORBIDDEN_MODULES = (
    "sqlite3",  # Fresh-process import probe confirmed it is not already loaded.
    "usearch",
    "tantivy",
    "torch",
    "sentence_transformers",
    "numba",
    "ranx",
    "fastmcp",
    "reflectlog.application.mcp_server",
    "reflectlog.application.memory.manager",
    "reflectlog.infrastructure.memory_store",
    "reflectlog.infrastructure.usearch_engine",
    "reflectlog.infrastructure.tantivy_engine",
    "reflectlog.infrastructure.embeddings",
)
HARNESS = """
import json, sys
from reflectlog.application.salvage.verification import ContentSalvageVerifier
result = ContentSalvageVerifier.from_defaults().verify(
    bundle=sys.argv[1], workspace_id=sys.argv[2]
)
print(result.status.value)
print("MODULES=" + json.dumps(sorted(sys.modules)))
"""


def _issues(result: VerificationResult) -> list[IssueTriple]:
    return [
        (issue.code.value, issue.entry.value, issue.line) for issue in result.issues
    ]


def _verify(bundle: Path) -> VerificationResult:
    return ContentSalvageVerifier.from_defaults().verify(
        bundle=str(bundle), workspace_id=WS
    )


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("verification crossed a forbidden I/O boundary")


def _require_symlinks(base: Path) -> None:
    """Probe both file and directory links, including Windows privilege failures."""
    target = base / "symlink-probe-target"
    target.mkdir()
    (target / "file").write_bytes(b"probe")
    try:
        (base / "symlink-probe-file").symlink_to(target / "file")
        (base / "symlink-probe-dir").symlink_to(target, target_is_directory=True)
    except OSError, NotImplementedError:
        pytest.skip("file/directory symlinks unavailable on this filesystem or account")


class CannedReader:
    def __init__(self, capture: BundleCapture) -> None:
        self.captured = capture
        self.calls: list[str] = []

    def capture(self, bundle_dir: str) -> BundleCapture:
        self.calls.append(bundle_dir)
        return self.captured


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling"])
def test_every_incomplete_entry_blocks_further_reads(tmp_path: Path, kind: str) -> None:
    if kind in {"symlink", "dangling"}:
        _require_symlinks(tmp_path)
    bundle = export_bundle(tmp_path)
    marker = bundle / "INCOMPLETE"
    match kind:
        case "file":
            marker.write_bytes(b"")
        case "directory":
            marker.mkdir()
        case "symlink":
            marker.symlink_to(bundle / "manifest.json")
        case "dangling":
            marker.symlink_to(tmp_path / "absent")
        case _:
            pytest.fail("unknown fixture kind")
    # A marker short-circuits even independent missing-file defects.
    (bundle / "memories.jsonl").unlink()

    result = _verify(bundle)

    assert result.status is VerificationStatus.INVALID
    assert _issues(result) == [("incomplete_marker_present", "marker", None)]
    assert result.record_count is None and result.rebuild_ready is None


@pytest.mark.parametrize("trailing_separator", [False, True])
def test_symlinked_root_is_rejected(tmp_path: Path, trailing_separator: bool) -> None:
    _require_symlinks(tmp_path)
    bundle = export_bundle(tmp_path)
    link = tmp_path / "linked-bundle"
    link.symlink_to(bundle, target_is_directory=True)

    result = ContentSalvageVerifier.from_defaults().verify(
        bundle=str(link) + (os.sep if trailing_separator else ""), workspace_id=WS
    )

    assert result.status is VerificationStatus.INVALID
    assert _issues(result) == [("bundle_symlink", "bundle", None)]


@pytest.mark.parametrize(
    "filename,entry", [("manifest.json", "manifest"), ("memories.jsonl", "records")]
)
@pytest.mark.parametrize("dangling", [False, True])
def test_required_file_symlinks_are_rejected(
    tmp_path: Path, filename: str, entry: str, dangling: bool
) -> None:
    _require_symlinks(tmp_path)
    bundle = export_bundle(tmp_path)
    required = bundle / filename
    target = tmp_path / "target"
    if not dangling:
        target.write_bytes(required.read_bytes())
    required.unlink()
    required.symlink_to(target)

    result = _verify(bundle)

    assert result.status is VerificationStatus.INVALID
    assert _issues(result) == [("file_symlink", entry, None)]


@pytest.mark.parametrize(
    "filename,entry", [("manifest.json", "manifest"), ("memories.jsonl", "records")]
)
@pytest.mark.parametrize("directory", [False, True], ids=["missing", "directory"])
def test_required_files_must_exist_and_be_regular(
    tmp_path: Path, filename: str, entry: str, directory: bool
) -> None:
    bundle = export_bundle(tmp_path)
    required = bundle / filename
    required.unlink()
    if directory:
        required.mkdir()

    result = _verify(bundle)

    assert result.status is VerificationStatus.INVALID
    assert _issues(result) == [
        ("file_not_regular" if directory else "file_missing", entry, None)
    ]


@pytest.mark.parametrize(
    "filename,entry", [("manifest.json", "manifest"), ("memories.jsonl", "records")]
)
def test_fifo_is_rejected_without_blocking(
    tmp_path: Path, filename: str, entry: str
) -> None:
    if "mkfifo" not in vars(os):
        pytest.skip("os.mkfifo is unavailable")
    bundle = export_bundle(tmp_path)
    required = bundle / filename
    required.unlink()
    try:
        os.mkfifo(required)
    except OSError, NotImplementedError:
        pytest.skip("FIFO creation unavailable on this filesystem or account")
    outcomes: Queue[VerificationResult | AssertionError | OSError] = Queue()

    def run() -> None:
        try:
            outcomes.put(_verify(bundle))
        except (AssertionError, OSError) as error:
            outcomes.put(error)

    worker = Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=2)

    assert not worker.is_alive(), "verifier blocked while opening a FIFO"
    result = outcomes.get_nowait()
    assert isinstance(result, VerificationResult), result
    assert result.status is VerificationStatus.INVALID
    assert _issues(result) == [("file_not_regular", entry, None)]


@pytest.mark.parametrize(
    "change,entry",
    [
        ("append", "records"),
        ("manifest", "manifest"),
        ("marker", "marker"),
        ("replace", "records"),
    ],
)
def test_observed_changes_invalidate_untrusted_results(
    tmp_path: Path, change: str, entry: str
) -> None:
    bundle = export_bundle(tmp_path)
    records = bundle / "memories.jsonl"
    replacement = tmp_path / "replacement"
    replacement.write_bytes(records.read_bytes())

    def mutate() -> None:
        match change:
            case "append":
                with records.open("ab") as handle:
                    _ = handle.write(record_line(3, "concurrent addition"))
            case "manifest":
                with (bundle / "manifest.json").open("ab") as handle:
                    _ = handle.write(b" ")
            case "marker":
                (bundle / "INCOMPLETE").write_bytes(b"")
            case "replace":
                os.replace(replacement, records)
            case _:
                pytest.fail("unknown change fixture")

    verifier = ContentSalvageVerifier(FileSystemBundleReader(after_capture=mutate))

    result = verifier.verify(bundle=str(bundle), workspace_id=WS)

    assert result.status is VerificationStatus.INVALID
    assert _issues(result) == [("changed_during_read", entry, None)]
    assert result.record_count is None
    assert result.rebuild_ready is None
    assert result.unresolved_reasons == ()


@pytest.mark.parametrize("case", ["valid", "invalid-record", "wrong-digest"])
def test_only_once_captured_bytes_decide_without_verifier_filesystem_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    bundle = export_bundle(tmp_path)
    manifest = load_manifest(bundle)
    data = record_line(7, "captured, not disk")
    expected: list[IssueTriple] = []
    if case == "invalid-record":
        data = record_line(7, "")
        expected = [("record_content_invalid", "records", 1)]
    records = as_object(manifest["records"])
    records["count"] = 1
    records["sha256"] = hashlib.sha256(data).hexdigest()
    if case == "wrong-digest":
        records["sha256"] = "0" * 64
        expected = [("digest_mismatch", "records", None)]
    reader = CannedReader(
        BundleCapture(
            CapturedFile(json.dumps(manifest).encode()), CapturedFile(data), ()
        )
    )
    protocol_reader: IBundleReader = reader
    verifier = ContentSalvageVerifier(protocol_reader)
    nonexistent = str(tmp_path / "never-created")

    with monkeypatch.context() as patcher:
        patcher.setattr(builtins, "open", _forbidden)
        patcher.setattr(os, "open", _forbidden)
        for method in ("read_bytes", "read_text", "stat"):
            patcher.setattr(Path, method, _forbidden)
        result = verifier.verify(bundle=nonexistent, workspace_id=WS)

    assert reader.calls == [nonexistent]
    assert result.status is (
        VerificationStatus.INVALID if expected else VerificationStatus.READY
    )
    assert _issues(result) == expected
    assert result.record_count == (None if case == "invalid-record" else 1)
    assert (bundle / "memories.jsonl").read_bytes() != data
    assert not Path(nonexistent).exists()


@pytest.mark.parametrize("state", ["ready", "not-ready", "corrupt", "missing-newline"])
def test_verification_never_writes_creates_scratch_opens_sqlite_or_repairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: BundleState
) -> None:
    journal = [(WS, "pending", "add", SECRET)] if state == "not-ready" else []
    bundle = export_bundle(tmp_path, journal=journal)
    expected: list[IssueTriple] = []
    match state:
        case "ready" | "not-ready":
            pass
        case "corrupt":
            write_records(bundle, b"{broken}\n")
            expected = [("json_invalid", "records", 1)]
        case "missing-newline":
            write_records(bundle, record_line(1, SECRET).removesuffix(b"\n"))
            expected = [("missing_final_newline", "records", 1)]
    before = tree_digest(bundle)
    parent_before = sorted(os.listdir(bundle.parent))
    bytes_before = (bundle / "memories.jsonl").read_bytes()
    verifier = ContentSalvageVerifier.from_defaults()

    with monkeypatch.context() as patcher:
        for name in (
            "mkstemp",
            "mkdtemp",
            "NamedTemporaryFile",
            "TemporaryDirectory",
            "SpooledTemporaryFile",
        ):
            patcher.setattr(tempfile, name, _forbidden)
        for name in (
            "mkdir",
            "makedirs",
            "rename",
            "replace",
            "remove",
            "unlink",
            "rmdir",
            "utime",
            "chmod",
            "truncate",
            "ftruncate",
            "fsync",
            "write",
        ):
            patcher.setattr(os, name, _forbidden)
        for name in (
            "copy",
            "copy2",
            "copyfile",
            "copyfileobj",
            "copymode",
            "copystat",
            "copytree",
            "move",
            "rmtree",
        ):
            patcher.setattr(shutil, name, _forbidden)
        patcher.setattr(sqlite3, "connect", _forbidden)
        patcher.setattr(subprocess, "Popen", _forbidden)
        result = verifier.verify(bundle=str(bundle), workspace_id=WS)

    expected_status = (
        VerificationStatus.INVALID
        if expected
        else (
            VerificationStatus.NOT_READY
            if state == "not-ready"
            else VerificationStatus.READY
        )
    )
    assert result.status is expected_status
    assert _issues(result) == expected
    assert tree_digest(bundle) == before
    assert sorted(os.listdir(bundle.parent)) == parent_before
    assert (bundle / "memories.jsonl").read_bytes() == bytes_before


def test_symlink_capability_failure_skips_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(*args: object, **kwargs: object) -> None:
        raise PermissionError("symlink privilege unavailable")

    monkeypatch.setattr(Path, "symlink_to", denied)

    with pytest.raises(pytest.skip.Exception):
        _require_symlinks(tmp_path)


@pytest.mark.parametrize(
    "record_path",
    [
        "../memories.jsonl",
        "/memories.jsonl",
        "nested/memories.jsonl",
        "C:\\memories.jsonl",
    ],
)
def test_unsafe_manifest_record_paths_are_rejected(
    tmp_path: Path, record_path: str
) -> None:
    bundle = export_bundle(tmp_path)
    manifest = load_manifest(bundle)
    as_object(manifest["records"])["file"] = record_path
    store_manifest(bundle, manifest)

    result = _verify(bundle)

    assert result.status is VerificationStatus.INVALID
    assert _issues(result) == [("records_path_invalid", "manifest", None)]


def test_verifier_imports_only_the_lazy_default_reader_from_infrastructure() -> None:
    tree = ast.parse(
        (REPO / "reflectlog/application/salvage/verification.py").read_text(
            encoding="utf-8"
        )
    )
    forbidden = (
        "sqlite3",
        "tempfile",
        "shutil",
        "subprocess",
        "socket",
        "usearch",
        "tantivy",
        "numba",
        "reflectlog.application.config",
        "reflectlog.application.mcp_server",
        "reflectlog.application.memory",
        "reflectlog.utility.scoring",
        "reflectlog.application.salvage.service",
    )
    imported: list[str] = []
    infrastructure: list[ast.Import | ast.ImportFrom] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [
                node.module,
                *(f"{node.module}.{alias.name}" for alias in node.names),
            ]
        imported.extend(names)
        if isinstance(node, ast.Import | ast.ImportFrom) and any(
            name.startswith("reflectlog.infrastructure") for name in names
        ):
            infrastructure.append(node)

    assert [
        name
        for name in imported
        if any(name == bad or name.startswith(f"{bad}.") for bad in forbidden)
    ] == []
    assert len(infrastructure) == 1
    reader_import = infrastructure[0]
    assert isinstance(reader_import, ast.ImportFrom)
    assert reader_import.module == "reflectlog.infrastructure.salvage_bundle_reader"
    assert [alias.name for alias in reader_import.names] == ["FileSystemBundleReader"]
    factories = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "from_defaults"
    ]
    assert len(factories) == 1
    assert reader_import in list(ast.walk(factories[0]))


@pytest.mark.parametrize("state", ["ready", "not-ready", "invalid"])
def test_isolated_import_and_default_verification_load_no_engines_or_sqlite(
    tmp_path: Path, state: str
) -> None:
    journal = [(WS, "pending", "add", SECRET)] if state == "not-ready" else []
    bundle = export_bundle(tmp_path, journal=journal)
    if state == "invalid":
        write_records(bundle, b"{broken}\n")

    completed = subprocess.run(
        [sys.executable, "-c", HARNESS, str(bundle), WS],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    status, module_line = completed.stdout.splitlines()
    assert status == state.replace("-", "_")
    modules: list[str] = json.loads(module_line.removeprefix("MODULES="))
    assert "reflectlog.infrastructure.salvage_bundle_reader" in modules
    assert (
        sorted(
            name
            for name in modules
            if any(
                name == bad or name.startswith(f"{bad}.") for bad in FORBIDDEN_MODULES
            )
        )
        == []
    )


@pytest.mark.parametrize(
    "content",
    ["tampered replacement", "x" * 500_001, "memory\0with NUL"],
    ids=["tampered", "over-add-limit", "nul"],
)
def test_consistency_is_neither_authenticity_nor_add_acceptance(
    tmp_path: Path, content: str
) -> None:
    bundle = export_bundle(tmp_path)
    # Recomputed hashes prove agreement, not origin. Format checks deliberately
    # do not invoke add validation (which rejects oversized batches and NULs).
    write_records(bundle, record_line(1, content), resync=True)

    result = _verify(bundle)

    assert result.status is VerificationStatus.READY
    assert _issues(result) == []
    assert result.record_count == 1
    assert result_payload(result)["authenticity"] == "not_verified"
